from PIL import Image
import numpy as np


# we need to read the heightmap inside ../terrain/lugano.png
# we need to convert from 16 bit to two 8 bit channels
# so the MSB (byte) goes to the red channel and the LSB (byte) goes to the green channel
# then we save the image in the same folder
def main():
    # load the heightmap
    heightmap = Image.open("../terrain/lugano.png")
    # convert the heightmap to a numpy array
    heightmap = np.array(heightmap)
    # split into two 8 bit channels
    heightmap_red = (heightmap >> 8).astype(np.uint8)  # Most significant byte
    heightmap_green = (heightmap & 0xFF).astype(np.uint8)  # Least significant byte
    # Combine channels into RGB image
    encoded_image = np.zeros(
        (heightmap.shape[0], heightmap.shape[1], 3), dtype=np.uint8
    )
    encoded_image[..., 0] = heightmap_red  # Red channel
    encoded_image[..., 1] = heightmap_green  # Green channel
    encoded_image[..., 2] = 0  # Unused blue channel
    # save the new image
    Image.fromarray(encoded_image).save("../terrain/lugano_8bit.png")


if __name__ == "__main__":
    main()
