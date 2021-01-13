import argparse
import numpy as np
import cv2


def make_pixel_image(pixels, colors, pixel_size=16, grid_thickness=2, grid_color=0):
    print("Recalculating pixel image (size %i, shape %s):  %f" % (pixel_size, pixels.shape,np.mean(pixels)))
    image = colors[pixels]
    image = np.repeat(image, pixel_size + grid_thickness, axis=0)
    image = np.repeat(image, pixel_size + grid_thickness, axis=1)

    for v in range(1, pixels.shape[1]):
        image[:, v * (pixel_size + grid_thickness) - grid_thickness: v * (pixel_size + grid_thickness), :] = grid_color
    for h in range(1, pixels.shape[0]):
        image[h * (pixel_size + grid_thickness) - grid_thickness: h * (pixel_size + grid_thickness), :, :] = grid_color

    image = image[:-grid_thickness, :-grid_thickness, :]
    if colors.shape[1] == 1:
        image = image[:, :, 0]
    return image


def test_make_pixel_image():
    test_in = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]]).astype(np.int64)

    test_colors = np.array([0, 1]).reshape(2, 1)

    test_out = np.array([[0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 1, 1, 1],
                         [0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 1, 1, 1],
                         [0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 1, 1, 1],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 0, 0, 0],
                         [0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 0, 0, 0],
                         [0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 0, 0, 0],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 1, 1, 1],
                         [1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 1, 1, 1],
                         [1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 1, 1, 1]])

    out = make_pixel_image(test_in, test_colors, pixel_size=3, grid_thickness=2, grid_color=2)
    assert np.array_equal(out, test_out)
    print("make_pixel_image test:  pass")


class CellEditor(object):
    COLORING_MODES = ['flip', 'selected']

    def __init__(self, w, h, colors=None):
        self._pixel_size = 32
        self._grid_size = 2
        self._width, self._height = w, h
        self._mode = self.COLORING_MODES[0]
        if colors is None:
            colors = [np.array((0xff, 0xfd, 0xd0)).astype(np.uint8),
                      np.array((0x00, 0x50, 0xff)).astype(np.uint8)]
        self._colors = np.array(colors)
        self._x = np.zeros((h, w)).astype(np.int64)
        self._finished = False
        self._disp_window_name = "Cell Editor"
        print("Cell Editor hotkeys: \nF to finish\n =/- magnify, \n o/p height pixels, \n[/] width pixels")

        cv2.namedWindow(self._disp_window_name)
        cv2.startWindowThread()
        cv2.setMouseCallback(self._disp_window_name, self._handle_mouse)

        self._update()

    def img_2_grid_coords(self, i, j):

        m = self._pixel_size + self._grid_size
        x = int(i / float(m))
        if (i % m) - self._pixel_size > 0:
            return None, None
        y = int(j / float(m))
        if (j % m) - self._pixel_size > 0:
            return None, None
        return x, y

    def _update(self):

        self._image = make_pixel_image(self._x, self._colors, self._pixel_size, self._grid_size)

    def get_cell(self):
        return self._x

    def _handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            row, col = self.img_2_grid_coords(y, x)
            if row is not None:
                self._change_pixel(row, col)

    def _change_pixel(self, row, col):
        print("Changing pixel at:  %s, %s" % (row, col))
        if self._mode == 'flip':
            self._x[row, col] = 1 - self._x[row, col]
        else:
            raise Exception("Coloring mode not implemented:  %s" % (self._mode,))
        self._update()

    def edit(self):
        def keymatch(k, x):
            if isinstance(x, list):
                return k in [ord(d) for d in x]
            return k == ord(x)

        while not self._finished:

            cv2.imshow(self._disp_window_name, self._image[:,:,::-1])

            key = cv2.waitKey(20) & 0xFF
            if keymatch(key, ['Q', 'q', 'F', 'f']):
                self._finished = True
                cv2.destroyWindow(self._disp_window_name)
                break
            elif keymatch(key, '-'):
                self._pixel_size = int(self._pixel_size * 0.8)
                self._pixel_size = 1 if self._pixel_size < 1 else self._pixel_size
                self._update()
            elif keymatch(key, '='):
                self._pixel_size = int(self._pixel_size * 1.2)
                self._update()
            elif keymatch(key, '['):
                self._width -= 1
                self._width = 1 if self._width < 1 else self._width
                self._x = self._x[:self._height, :self._width]
                self._update()
            elif keymatch(key, ']'):
                self._width += 1
                self._x = np.hstack((self._x, np.zeros(self._x.shape[0]).reshape(-1,1))).astype(np.int64)
                self._update()
            elif keymatch(key, ['o', 'O']):
                self._height -= 1
                self._height = 1 if self._height < 1 else self._height
                self._x = self._x[:self._height, :self._width]
                self._update()
            elif keymatch(key, ['P', 'p']):
                self._height += 1
                self._x = np.vstack((self._x, np.zeros(self._x.shape[1]).reshape(1, -1))).astype(np.int64)
                self._update()
            elif not key == 0xFF:
                print("Unknown keypress:  %s" % (key,))


def cell_editor_test():
    """
    interactive UI test
    """
    ce = CellEditor(10, 7)
    ce.edit()
    result = ce.get_cell()
    print("Result:  \n%s" % (result,))


if __name__ == "__main__":
    test_make_pixel_image()
    cell_editor_test()
