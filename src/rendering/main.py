import sys
import pygame
import time
import numpy as np

from rendering.sprites import *
from rendering.config import *
from encoder import StaghuntEncoder

class StaghuntRenderer:
    def __init__(self, state, autonomous=False, encoder=StaghuntEncoder()):
        pygame.init()

        # if the staghunt environment is controlling the characters
        self.auto = autonomous

        self.enc = encoder

        self.tilesize = 32
        self.WIN_WIDTH = 640
        self.WIN_HEIGHT = 480
        self.FPS = 60

        self.speeds = { "PLAYER_SPEED" : 8, "ENEMY_SPEED" : 2, "CHARACTER_SPEED" : 32}
        self.layers = {"PLAYER_LAYER" : 4, "ENEMY_LAYER" : 3, "BLOCK_LAYER" : 2, "GROUND_LAYER" : 1 }
        self.colors = {"WHITE" : (255, 255, 255), "BLACK" : (0, 0, 0), "RED" : (255, 0, 0), "BLUE" : (0, 0, 255) }

        self.screen = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.Font('./rendering/verdana/verdana.ttf', 32)

        self.character_spritesheet = Spritesheet('./rendering/img/character.png')
        self.terrain_spritesheet = Spritesheet('./rendering/img/terrain.png')
        self.enemy_spritesheet = Spritesheet('./rendering/img/enemy.png')
        self.attack_spritesheet = Spritesheet('./rendering/img/attack.png')

        self.state = state
        self.state_history = [state]
        '''@TODO: Allow init without state, use default state'''
        self.init_tilemap(self.state.map)

    def init_tilemap(self, tilemap):
        if tilemap is not None:
            self.tilemap = tilemap
            self.base_map = tilemap
        else:
            map = np.zeros((7, 7))
            for i in range(7):
                for j in range(7):
                    map[i][j] = 1
            self.tilemap = map
            self.base_map = map

    def create_tilemap(self, tilemap=None):
        tilemap = self.tilemap if tilemap is None else tilemap
        for i, row in enumerate(tilemap):
            for j, column in enumerate(row):
                Ground(self, j, i)
                if column == 0:
                    Block(self, j, i)
                elif column > 0:
                    ids = self.enc.decode_id(column)
                    for id in ids:
                        if id == "h1":
                            self.player = Player(self, j, i, id=id)
                        else:
                            type = id[0]
                            if type == "s" or type == "r" or type == "h":
                                Enemy(self, j, i, id=id)

    def draw(self):
        self.screen.fill(self.colors["BLACK"])
        self.all_sprites.draw(self.screen)
        pygame.display.update()

    def update(self):
        # find the update method in every sprite and call it
        self.all_sprites.update()

    def new_game(self):
        self.playing = True
        # Contains all objects in the game (e.g. player, enemies, walls)
        self.all_sprites = pygame.sprite.LayeredUpdates()
        self.blocks = pygame.sprite.LayeredUpdates()
        self.enemies = pygame.sprite.LayeredUpdates()
        self.attacks = pygame.sprite.LayeredUpdates()
        self.create_tilemap()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.playing = False
                self.running = False

    def main(self):
         while self.playing:
             self.events()
             self.update()
             self.draw()
         self.running = False

    def run(self):
        self.new_game()
        while self.running:
            self.main()
        pygame.quit()
        sys.exit()

    ''' Auto Control Methods '''
    def visualize(self):
        self.new_game()
        self.step()

    def step(self, state=None):
        if state is not None:
            self.state = state
        self.events()
        speed = TRANSITION_SPEED if len(self.state_history) > 1 else 1
        for i in range(speed):
            print("\nt", i)
            self.update()
            self.draw()
            if speed != 1:
                time.sleep(2/speed)
        self.state_history.append(self.state)

    def stop(self):
        pygame.quit()
        sys.exit()

    def reset(self, state):
        self.state = state
        self.state_history = [state]
        self.init_tilemap(self.state.map)
        self.new_game()
        self.visualize()

def main():
    g = StaghuntRenderer()
    g.run()

if __name__ == '__main__':
    main()
