# Link - https://qq.readthedocs.io/en/latest/tiles.html

import pygame
import pygame.locals


def load_title_table(filename, width, height):
    pass


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((128, 98))
    screen.fill((255, 255, 255))
    table = load_title_table("pgame-tiled-tileset.png", 24, 16)

    for x, row in enumerate(table):
        for y, tile in enumerate(row):
            screen.blit(tile, (x * 32, y * 24))

    pygame.display.flip()

    while pygame.event.wait().type != pygame.locals.QUIT:
        pass
