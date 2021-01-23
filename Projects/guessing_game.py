import random

# Guess by user input
def guess(x):
    random_number = random.randint(0, x)
    guess = 0
    while guess != random_number:
        guess = int(input(f'Guess a number between 0 and {x}: '))

        if guess < random_number:
            print("Sorry, guess again. Too low")
        elif guess > random_number:
            print('Sorry, guess again. Too high')

    print(f'Yay, you have guessed the correct number')


def computer_guess(x):
    low = 0
    high = x
    feedback = ''
    guess = ''

    while feedback != 'c':

        if low != high:
            guess = random.randint(low, high)
        else:
            guess = low

        feedback = input(f'Is {guess} too high (H), too low (L), or correct (C)??'.lower())
        if feedback == 'h':
            high = guess - 1
        if feedback == 'l':
            low = guess + 1

    print(f'Yay! The computer guessed your number, {x}, correctly!')

computer_guess(1000)