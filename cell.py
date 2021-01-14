"""Gui widget for exiting a pixel grid """
import argparse
import numpy as np
import cv2
from threading import Lock, Thread
from util import COLORS, make_pixel_image, keymatch


class CellEditor(object):
    COLORING_MODES = ['flip', 'selected']
    DEFAULT_HOTKEYS = {'res_decrease': '-',
                       'res_increase': '=',
                       'width_decrease': '[',
                       'width_increase': ']',
                       'height_decrease': ['o', 'O'],
                       'height_increase': ['p', 'P'],
                       'spacer_width_decrease': ';',
                       'spacer_width_increase': "'"}


    def __init__(self, w=3, h=3, colors=None, change_callback=None, asynch=False, name=None, constraints=None,
                 hotkeys=None):
        """
        Gui widget for editing a pixel grid ("pattern")
        :param w:  initial width, pixels
        :param h:  height
        :param colors:  2x3 array, 2 RGB colors, (named "0" and "1")
        :param change_callback:  Call this function (with new pattern, name as args) when pattern changes
        :param asynch:  this constructor will start the editor in a thread and return
        :param name:  for keeping track
        :param constraints: dict with constraints:
            {'w': width constraint, can be: int - hold constant at this value
                                            (lo, hi) - two ints, stay in this range, value(s) can be None for 1-sided
                                            CellEditor() - keep locked to this editor's width,.  When it changes, this
                                                object will change it's width to match, and vice versa.

                                            Cell Editors can only be paired, using a CE that is already part of another
                                            pairing's constraint will remove that pairing (and constraint).
             'h': height constraints, same format.}
         :param hotkeys: dict defining hotkeys, see DEFAULT_HOTKEYS for options

        """
        print("New Cell created:  w=%i, h=%i" % (w, h))
        print("\tCommands - <> adjust width, .\n")
        self._pixel_size = 32
        self._grid_size = 2
        self._callback = change_callback
        self._hotkeys = self.DEFAULT_HOTKEYS if hotkeys is None else hotkeys
        self._width, self._height = w, h
        self._mode = self.COLORING_MODES[0]
        if colors is None:
            colors = [COLORS['cream'], COLORS['sky_blue']]
        self._colors = np.array(colors)
        self._x = np.zeros((h, w)).astype(np.int64)
        self._finished = False
        self._name = name
        self._constraints = constraints if constraints is not None else {}
        for k in self._constraints:
            self._constraints[k].set_constraint(k, self)
        self._disp_window_name = "Cell Editor - %s" % (name,)
        print("Cell Editor hotkeys: \n\tF to finish\n\t=- magnify, \n\top height pixels, \n\t[] width pixels\n\t;' spacer height\n")

        cv2.namedWindow(self._disp_window_name)
        cv2.startWindowThread()
        cv2.setMouseCallback(self._disp_window_name, self._handle_mouse)
        self._mutex = Lock()
        self._update()

        self._asynch = asynch
        if asynch:
            self.edit(asynch=True)

    def set_constraint(self, dim, c):
        self._constraints[dim] = c

    def img_2_grid_coords(self, i, j):

        m = self._pixel_size + self._grid_size
        x = int(i / float(m))
        if (i % m) - self._pixel_size > 0:
            return None, None
        y = int(j / float(m))
        if (j % m) - self._pixel_size > 0:
            return None, None
        return x, y

    def _update(self, no_callback=False, no_regenerate=False):
        if not no_regenerate:
            print(self._x)
            self._image = make_pixel_image(self._x, self._colors, self._pixel_size, self._grid_size)
        if self._callback is not None and not no_callback:
            self._callback(self._x, self._name)

    def get_cell(self):
        with self._mutex:
            return self._x

    def _handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            row, col = self.img_2_grid_coords(y, x)
            if row is not None:
                self._change_pixel(row, col)

    def _change_pixel(self, row, col):
        if self._mode == 'flip':
            with self._mutex:
                self._x[row, col] = 1 - self._x[row, col]
        else:
            raise Exception("Coloring mode not implemented:  %s" % (self._mode,))
        self._update()

    def is_finished(self):
        return self._finished

    def finish(self):
        self._finished = True

    def edit(self, asynch=False):
        if asynch:
            t = Thread(target=self._start_editing)
            t.start()
            return t
        else:
            return self._start_editing()

    def dim_is_locked(self, dim, marked=None):
        """
        Chan change grid dimension?
        :param dim: "w" or "h"
        :return:  True if indicated dimension is changeable, False if locked
        """
        marked = [] if marked is None else marked
        if dim in self._constraints:
            if isinstance(self._constraints[dim], int):
                return True
            if self._constraints[dim] not in marked:
                marked.append(self._constraints[dim])
                return self._constraints[dim].dim_is_locked(dim, marked = marked)
        return False

    def __del__(self):
        """
        NEED TO RE-CONNECT CONSTRAINT CHAIN HERE!
        :return:
        """
        pass

    def change_constraint(self, dim, constraint):
        self._constraints[dim] = constraint

    def change_dims(self, dim=None, inc=0, depth=1):
        """
        Grow/shrink number of pixels.
        Will not violate constrains.
        Will propagate changes to constrained CellEditors.

        :param dim: string, must be 'w' or 'h'
        :param inc:  int, change to make, i.e. 2 means add 2 pixel rows, etc.
        :param depth:  limit constraint propagation recursion
        """
        with self._mutex:

            if not self.dim_is_locked(dim):
                if inc < 0:  # shrink

                    if dim == "w":
                        self._width += inc
                        self._width = 1 if self._width < 1 else self._width
                        self._x = self._x[:self._height, :self._width]
                    elif dim == "h":
                        self._height += inc
                        self._height = 1 if self._height < 1 else self._height
                        if inc < 1:  # shrink
                            self._x = self._x[:self._height, :self._width]

                elif inc > 0:  # grow
                    if dim == 'w':
                        self._width += inc
                        self._x = np.hstack((self._x, np.zeros((self._x.shape[0], 1), dtype=self._x.dtype)))
                    elif dim == 'h':
                        self._height += inc
                        self._x = np.vstack((self._x, np.zeros((1, self._x.shape[1]), dtype=self._x.dtype)))
                if dim in self._constraints and isinstance(self._constraints[dim], CellEditor) and depth > 0:
                    self._constraints[dim].change_dims(dim, inc, depth - 1)
        print("%s - CD done - w=%i, h=%i, x.shape=%s" % (self._name, self._width, self._height, self._x.shape))

        self._update()

    def update_hotkeys(self, updates):
        self._hotkeys.update(updates)

    def _start_editing(self):

        while not self._finished:

            cv2.imshow(self._disp_window_name, self._image[:, :, ::-1])

            key = cv2.waitKey(20) & 0xFF
            if keymatch(key, ['Q', 'q', 'F', 'f']):
                self._finished = True
                cv2.destroyWindow(self._disp_window_name)
                break
            elif keymatch(key, self._hotkeys['res_decrease']):
                self._pixel_size = int(self._pixel_size * 0.75)
                self._pixel_size = 4 if self._pixel_size < 4 else self._pixel_size
                self._update(no_regenerate=True)  # just changes display
            elif keymatch(key, self._hotkeys['res_increase']):
                self._pixel_size = int(self._pixel_size * 1.25)
                self._pixel_size = 4 if self._pixel_size < 4 else self._pixel_size  # don't get too small!
                self._update(no_regenerate=True)

            elif keymatch(key, self._hotkeys['width_decrease']):
                if self._name != "spacer_pattern":
                    self.change_dims('w', -1)
            elif keymatch(key, self._hotkeys['width_increase']):
                if self._name != "spacer_pattern":
                    self.change_dims('w', 1)

            elif keymatch(key, self._hotkeys['height_decrease']):
                if self._name != "spacer_pattern":
                    self.change_dims('h', -1)
            elif keymatch(key, self._hotkeys['height_increase']):
                if self._name != "spacer_pattern":
                    self.change_dims('h', 1)

            elif keymatch(key, self._hotkeys['spacer_width_decrease']):
                if self._name =="spacer_pattern":
                    self.change_dims('w', -1)
            elif keymatch(key, self._hotkeys['spacer_width_increase']):
                if self._name == "spacer_pattern":
                    self.change_dims('w', 1)



            elif not key == 0xFF:
                print("Unknown keypress:  %s" % (key,))
        cv2.destroyWindow(self._disp_window_name)
        print("%s - Ended main loop." % (self._name, ))


def cell_editor_test():
    """
    interactive UI test
    """
    ce = CellEditor(10, 7)
    ce.edit()
    result = ce.get_cell()
    print("Result:  \n%s" % (result,))


if __name__ == "__main__":
    cell_editor_test()
