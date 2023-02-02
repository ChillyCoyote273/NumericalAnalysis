import pygame
import numpy as np


class Robot:
	def __init__(self):
		self.pose = np.array([0, 0, 0])
		self.corners = np.array([
			[16, 16],
			[16, -16],
			[-16, 16],
			[-16, -16]
		])

	def draw(self, surface):
		pass


def inches_to_pixels(x):
	return x * 4


def main():
	pygame.init()
	screen = pygame.display.set_mode((580, 580))
	clock = pygame.time.Clock()

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				raise SystemExit
		
		screen.fill('dark gray')

		for i in range(6):
			for j in range(6):
				tile = pygame.rect.Rect((97 * i, 97 * j), (95, 95))
				pygame.draw.rect(screen, 'light gray', tile)

		pygame.display.flip()
		clock.tick(60)


if __name__ == '__main__':
	main()
