import pygame
import carla

from agents.navigation.basic_agent import BasicAgent

def main():
    pygame.init()
    pygame.font.init()

    word=None
    seed=100
    random.seed(seed)

    host='127.0.0.1'
    port=2000
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()

    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('CARLA Test')
    pass

if __name__ == '__main__':
    main()
