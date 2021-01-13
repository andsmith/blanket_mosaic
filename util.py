import numpy as np


COLORS = {'cream': np.array((0xff, 0xfd, 0xd0)).astype(np.uint8),
          'sky_blue': np.array((0x00, 0x50, 0xff)).astype(np.uint8),
          'neon_green': np.array([130, 255, 120], dtype=np.uint8),
          'green': np.array([0, 255, 0], dtype=np.uint8),
          'dk_green': np.array([0, 100, 0], dtype=np.uint8)}


def make_pixel_image(pixels, colors, pixel_size=16, grid_thickness=2, grid_color=0):
    image = colors[pixels]
    image = np.repeat(image, pixel_size + grid_thickness, axis=0)
    image = np.repeat(image, pixel_size + grid_thickness, axis=1)

    if grid_thickness > 0:
        for v in range(1, pixels.shape[1]):
            image[:, v * (pixel_size + grid_thickness) - grid_thickness: v * (pixel_size + grid_thickness),
            :] = grid_color
        for h in range(1, pixels.shape[0]):
            image[h * (pixel_size + grid_thickness) - grid_thickness: h * (pixel_size + grid_thickness), :,
            :] = grid_color
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


def keymatch(k, x):
    if isinstance(x, list):
        return k in [ord(d) for d in x]
    return k == ord(x)
