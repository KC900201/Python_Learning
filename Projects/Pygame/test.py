import pygame


def main():
    # initialize pygame module
    pygame.init()
    # load and set logo
    pygame.display.set_caption("minimal program")

    # create a surface on screen that has the size of 240 * 240
    screen = pygame.display.set_mode((240, 180))

    # define
    running = True

    while running:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()
