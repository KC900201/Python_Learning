# Directory - https://www.pygame.org/docs/tut/ChimpLineByLine.html

# Import Modules
import os
import pygame as pg

if not pg.font:
    print("Fonts disabled")

if not pg.mixer:
    print("Sound disabled")

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")


# Functions
def load_image(name, colorkey=None, scale=1):
    fullname = os.path.join(data_dir, name)
    image = pg.image.load(fullname)

    size = image.get_size()
    size = (size[0] * scale, size[1] * scale)
    image = pg.transform.scale(image, size)

    image = image.convert()

    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return image, image.get_rect()


def load_sound(name):
    class NoneSound:
        def play(self):
            pass

    if not pg.mixer or not pg.mixer.get_init():
        return NoneSound()

    fullname = os.path.join(data_dir, name)
    sound = pg.mixer.Sound(fullname)

    return sound


# Game Object classes
class Fist(pg.sprite.Sprite):
    """move a clenched fist on the screen, following the mouse"""

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image("fist.png", 1)
        self.fist_offset = (-235, -80)
        self.punching = False

    # Continue
    def update(self):
        pass

    def punch(self):
        pass

    def unpunch(self):
        pass


class Chimp(pg.sprite.Sprite):

    def __init__(self):
        pass

    def update(self):
        pass

    def _walk(self):
        pass

    def _spin(self):
        pass

    def punched(self):
        pass


if __name__ == '__main__':
    pg.init()
