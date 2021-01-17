"""
Created on Sun March 08 2020

@author: Kwong Cheong Ng
@filename: pygame_tutorial.py
@coding: utf-8
========================
Date          Comment
========================
03082020      First revision 
03162020
"""
import pygame

# Set up parameters
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

if __name__ == '__main__':

    pygame.init()
    
    # Set up pygame display parameters 
    gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT)) #width, 
    pygame.display.set_caption('Hello')
    clock = pygame.time.Clock()

    black = (0, 0, 0)
    white = (255, 255, 255)    
    red = (255, 0, 0)

    car_width = 73

    # Pygame clock mode
    crashed = False
    carImg = pygame.image.load('pygame/racecar.png')

    def car(x, y):
        gameDisplay.blit(carImg, (x,y))

    def game_loop():
        x = (DISPLAY_WIDTH * 0.50)
        y = (DISPLAY_HEIGHT * 0.50)
        x_change = 0
        car_speed = 0
        
        gameExit = False

        
        while not gameExit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
        
                # Add move keys event
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x_change = -5
                    elif event.key == pygame.K_RIGHT:
                        x_change = 5
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        x_change = 0
                # End add move keys event
                #print(event)
            ## increment x
            x += x_change 
       
            gameDisplay.fill(white)
            car(x, y)        

            if x > DISPLAY_WIDTH - car_width or x < 0:
                gameExit = True

            pygame.display.update()
            clock.tick(60)

    game_loop()
    pygame.quit()
    quit()


