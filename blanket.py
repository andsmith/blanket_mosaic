import os
import argparse
import numpy as np
import cv2
from threading import Thread
from cell import CellEditor

from util import COLORS, make_pixel_image, keymatch


class PatternCell(object):
    def __init__(self, x, origin, spacer=None):
        self.x = x
        self.origin = origin
        self.spacer = spacer


class BlanketEditor(object):
    SYMMETRIES = ('rotational', 'translational')

    def __init__(self, w, h, symmetry, colors=None):
        if symmetry not in self.SYMMETRIES:
            raise Exception("symmetry must be on eof %s" % (self.SYMMETRIES,))
        self._row_parity = 0
        self._w, self._h = w, h
        self._finished = False
        self._cells = []
        self._colors = colors if colors is not None else np.array([COLORS['cream'], COLORS['sky_blue']])
        self._px_size = 4
        self._box_thickness = 1
        self._rotation_direction = 1
        self._window_name = "mosaic blanket!"
        cv2.namedWindow(self._window_name)
        self._cell_editor = None
        self._spacer_editor = None
        self._pos = (0, 0)  # where next cell goes
        self._current_pattern = PatternCell(None, origin=np.array(self._pos))

        # blanket & w/annotations
        self._image = None
        self._display_image = None  # same here

    def _update_image(self):  # update current blanket
        if self._current_pattern.x is None:  # when starting
            print("WARN - update_image called with no pattern")
            return

        x = np.zeros((self._h, self._w), dtype=np.int64)

        # add cell rows
        print("Cells:\n\t%s" % ("\n\t".join(["%s, %s" % (c.x.shape, c.origin) for c in self._cells+ [self._current_pattern]]), ))
        for c in self._cells + [self._current_pattern]:

            # draw horizontal:

            spacer = c.spacer if c.spacer is not None else np.zeros((c.x.shape[0], 0))
            spacer_vert = np.rot90(spacer, k=1)
            cx_vert = np.rot90(c.x, k=1)

            n_chunks_w = int((self._w - c.origin[1] * 2.0 - c.x.shape[0] - spacer.shape[1]) / c.x.shape[1])
            n_chunks_h = int((self._h - c.origin[0] * 2.0 - c.x.shape[0] - spacer.shape[1]) / c.x.shape[1])
            for w_chunk in range(n_chunks_w):
                row = c.origin[0]
                col = c.origin[1] + w_chunk * c.x.shape[1]
                if spacer.size > 0 and w_chunk == n_chunks_w - 1:  # corner piece
                    s_row = row
                    s_col = col + c.x.shape[1]
                    x[s_row:s_row + spacer.shape[0], s_col:s_col + spacer.shape[1]] = spacer
                x[row:row + c.x.shape[0], col:col + c.x.shape[1]] = c.x

            for h_chunk in range(n_chunks_h):
                row = c.origin[0] + c.x.shape[0] + h_chunk * cx_vert.shape[0]
                col = c.origin[1]
                if spacer.size > 0:
                    row += spacer_vert.shape[0]
                    if h_chunk == 0:

                        x[c.origin[0] + c.x.shape[0]: c.origin[0] + c.x.shape[0] + spacer_vert.shape[0],
                          c.origin[1]: c.origin[1] + spacer_vert.shape[1]] = spacer_vert
                x[row: row + cx_vert.shape[0], col:col + cx_vert.shape[1]] = cx_vert

        # make into viewable image

        new_image = make_pixel_image(x, self._colors, self._px_size, 0)

        # with self._image_lock:
        self._image = new_image

        # force to regenerate
        self._display_image = None

    def _refresh(self):  # update onscreen image and/or annotations
        if self._image is None:
            self._update_image()
        if self._image is not None:
            image = self._image.copy()
            if self._current_pattern.x is not None:
                px = self._current_pattern.x
                # annotate with current cell box
                p0 = np.array(self._pos) * self._px_size
                p1 = p0 + np.array(px.shape) * self._px_size
                image = cv2.rectangle(image, tuple(p0[::-1].tolist()), tuple(p1[::-1].tolist()),
                                      COLORS['dk_green'].tolist(), thickness=self._box_thickness)
            # with self._dimage_lock:
            self._display_image = image

    def _update_pattern_callback(self, new_cell_data, name):

        if name is not None:
            """if self._current_pattern is None:  # before it is connected, mark for refresh, but do nothing yet
                self._image = None
                self._display_image = None
                return"""
            if name == 'main_pattern':
                self._current_pattern.x = new_cell_data
            elif name == 'spacer_pattern':
                self._current_pattern.spacer = new_cell_data
            else:
                raise Exception("Pattern update called from thread with unrecognized name:  %s" % (name,))

            self._image = None
            self._display_image = None
        else:
            raise Exception("Update called from unnamed thread!")

    def edit(self):
        while not self._finished:

            # Change state
            if self._cell_editor is None:  # signal to start a new layer of cells
                shape = self._cells[-1].x.shape if len(self._cells) > 0 else (2, 2)

                self._cell_editor = CellEditor(w=shape[1], h=shape[0], colors=self._colors,
                                               change_callback=self._update_pattern_callback,
                                               asynch=True, name='main_pattern')

                self._image = None
                self._display_image = None

            elif self._cell_editor.is_finished():  # signal to finish current layer
                # accumulate

                self._cells.append(self._current_pattern)

                if self._spacer_editor is not None:
                    self._spacer_editor.finish()
                    self._spacer_editor = None

                self._cell_editor.finish()
                self._cell_editor = None

                # advance, using last used shape
                self._pos += np.array(self._cells[-1].x.shape)

                # set up for next cell editor
                self._current_pattern = PatternCell(x=None, origin=self._pos.copy())

            # updated display
            # with self._dimage_lock:  # prob. unnecessary
            if self._display_image is None:

                self._refresh()

            if self._display_image is not None:
                cv2.imshow(self._window_name, self._display_image[:, :, ::-1])

            # process keyboard
            key = cv2.waitKey(20) & 0xff
            if keymatch(key, ['Q', 'q']):
                self.save()
                if self._cell_editor is not None:
                    self._cell_editor.finish()
                break
            if keymatch(key, ['S', 's']):
                self.save()

            if keymatch(key, ' '):

                if self._spacer_editor is None and self._cell_editor is not None:
                    # Add a spacer, if not already existing.
                    self._spacer_editor = CellEditor(w=1,
                                                     h=self._current_pattern.x.shape[0],
                                                     colors=self._colors,
                                                     change_callback=self._update_pattern_callback,
                                                     asynch=True, name='spacer_pattern',
                                                     constraints={'h': self._cell_editor})

                else:
                    # Otherwise disconnect & delete it

                    self._spacer_editor.finish()
                    self._current_pattern.spacer=None
                    self._display_image = None
                    self._image = None

            elif keymatch(key, ['A', 'a']):
                self._px_size = int(self._px_size * 0.75)
                self._px_size = 4 if self._px_size < 4 else self._px_size
                self._display_image = None
                self._image = None
            elif keymatch(key, ['Z', 'z']):
                self._px_size = int(self._px_size * 1.25)
                self._px_size = 4 if self._px_size < 4 else self._px_size  # don't get too small
                self._display_image = None
                self._image = None

    def save(self):
        filenum = 0
        while True:
            filename = "blanket_%.4i.png" % (filenum,)
            if os.path.exists(filename):
                filenum += 1
            else:
                break
        cv2.imwrite(filename, self._display_image[:, :, ::-1])
        print("Wrote:  %s" % (filename,))


def setup():
    parser = argparse.ArgumentParser(description='Design blanket pattern.')
    parser.add_argument('--symmetry', '-s', type=str,
                        help="Blanket symmetry, must be one of:  %s." % (", ".join(BlanketEditor.SYMMETRIES),),
                        default='rotational')
    parser.add_argument('--width', '-w', type=float, help="Width of blanket (total stitches)", default=64)
    parser.add_argument('--height', '-t', type=float, help="Width of blanket (total stitches)", default=64)
    parsed = parser.parse_args()

    b = BlanketEditor(parsed.width, parsed.height, parsed.symmetry)
    return b


if __name__ == "__main__":
    b = setup()
    b.edit()
