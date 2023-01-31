import math
import random
import pygame

from rendering.config import *

class Spritesheet():
    def __init__(self, file):
        self.sheet = pygame.image.load(file).convert()

    def get_sprite(self, x, y, width, height):
        sprite = pygame.Surface([width, height])
        sprite.blit(self.sheet, (0, 0), (x, y, width, height)) # 3rd param gives cutout
        sprite.set_colorkey((0, 0, 0))
        return sprite

''' Static Game Objects '''

class StaticObject(pygame.sprite.Sprite):
    def __init__(self, game, x, y, layer=1, additional_groups=None, image=None, offset=True):
        self.game = game
        self._layer = layer
        self.groups = self.game.all_sprites
        if additional_groups is not None:
            self.groups = self.game.all_sprites, additional_groups
        pygame.sprite.Sprite.__init__(self, self.groups)

        position_factor = game.tilesize if offset else 1
        self.x = x * position_factor
        self.y = y * position_factor
        self.width = game.tilesize
        self.height = game.tilesize

    ''' Override Methods '''

    def set_image(self, image):
        if image is not None:
            self.image = image
            self.rect = self.image.get_rect()
            self.rect.x = self.x
            self.rect.y = self.y
        else:
            print("E: Recieved empty image.")

class Ground(StaticObject):
    def __init__(self, game, x, y):
        StaticObject.__init__(self, game, x, y, layer=game.layers["GROUND_LAYER"])
        image = game.terrain_spritesheet.get_sprite(64, 352, self.width, self.height)
        self.set_image(image)

class Block(StaticObject):
    def __init__(self, game, x, y):
        StaticObject.__init__(self, game, x, y, layer=game.layers["BLOCK_LAYER"], additional_groups=game.blocks)
        image = game.terrain_spritesheet.get_sprite(960, 448, self.width, self.height)
        self.set_image(image)

class DynamicObject(StaticObject):
    def __init__(self, game, x, y, layer=1, additional_groups=None, image=None, offset=1):
        StaticObject.__init__(self, game, x, y, layer=layer, additional_groups=additional_groups, image=image, offset=offset)
        image = game.terrain_spritesheet.get_sprite(960, 448, self.width, self.height)
        self.set_image(image)

        self.direction = 'down'

        self.animation_loop = 0
        self.set_animations()

    def update(self):
        self.animate()
        self.collide()

    def collide(self):
        pass

    def animate(self):
        self.direction = self.game.player.facing
        if self.animations is not None and self.direction in ["up", "down", "left", "right"]:
            self.image = self.animations[self.direction][math.floor(self.animation_loop)]
            self.animation_loop += 0.5
            if self.animation_loop >= 5:
                self.animation_loop = 0
                self.kill()

class Attack(DynamicObject):
    def __init__(self, game, x, y):
        DynamicObject.__init__(self, game, x, y, layer=game.layers["PLAYER_LAYER"], additional_groups=game.attacks, offset=0)
        image = game.attack_spritesheet.get_sprite(0, 0, self.width, self.height)
        self.set_image(image)

    def collide(self):
        hits = pygame.sprite.spritecollide(self, self.game.enemies, True)

    def set_animations(self):
        right_animations = [self.game.attack_spritesheet.get_sprite(0, 64, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(32, 64, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(64, 64, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(96, 64, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(128, 64, self.width, self.height)]

        down_animations = [self.game.attack_spritesheet.get_sprite(0, 32, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(32, 32, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(64, 32, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(96, 32, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(128, 32, self.width, self.height)]

        left_animations = [self.game.attack_spritesheet.get_sprite(0, 96, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(32, 96, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(64, 96, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(96, 96, self.width, self.height),
                           self.game.attack_spritesheet.get_sprite(128, 96, self.width, self.height)]

        up_animations = [self.game.attack_spritesheet.get_sprite(0, 0, self.width, self.height),
                         self.game.attack_spritesheet.get_sprite(32, 0, self.width, self.height),
                         self.game.attack_spritesheet.get_sprite(64, 0, self.width, self.height),
                         self.game.attack_spritesheet.get_sprite(96, 0, self.width, self.height),
                         self.game.attack_spritesheet.get_sprite(128, 0, self.width, self.height)]

        self.animations = {
            "down" : down_animations,
            "up" : up_animations,
            "left" : left_animations,
            "right" : right_animations
        }

''' Mobile Game Objects '''

class Character(DynamicObject):
    def __init__(self, game, x, y, id="U0", layer=1, additional_groups=None, image=None):
        DynamicObject.__init__(self, game, x, y, layer=layer, additional_groups=additional_groups, image=image)

        self.id = id

        self.transitioning = False
        self.transition_loop = 0
        self.movement_loop = 0
        self.max_travel = 5

        self.x_change = 0
        self.y_change = 0
        self.facing = 'down'

        self.set_image(self.image) # remove black background for characters

    def set_image(self, image):
        if image is not None:
            self.image = image
            self.image.set_colorkey(self.game.colors["BLACK"])

            self.rect = self.image.get_rect()
            self.rect.x = self.x
            self.rect.y = self.y
        else:
            print("E: Recieved empty image.")

    def animate(self):
        if self.animations is not None and self.facing in ["up", "down", "left", "right"]:
            if (self.facing in ["up", "down"] and self.y_change == 0) or (self.facing in ["left", "right"] and self.x_change == 0):
                self.image = self.animations[self.facing][0]
            else:
                self.image = self.animations[self.facing][math.floor(self.animation_loop)]
                self.animation_loop += 1 # 0.1
                if self.animation_loop >= 3:
                    self.animation_loop = 0 #1

    def collide_blocks(self, direction):
        if direction == "x":
            hits = pygame.sprite.spritecollide(self, self.game.blocks, False) # checking if the rect of two sprites are overlapping, False = don't delete sprite when collide
            if hits:
                if self.x_change > 0:
                    self.rect.x = hits[0].rect.left - self.rect.width # hits[0] = wall we're colliding with
                    self.facing = 'left'
                if self.x_change < 0:
                    self.rect.x = hits[0].rect.right
                    self.facing = 'right'
        if direction == "y":
            hits = pygame.sprite.spritecollide(self, self.game.blocks, False)
            if hits:
                if self.y_change > 0:
                    self.rect.y = hits[0].rect.top - self.rect.height
                    self.facing = 'up'
                if self.y_change < 0:
                    self.rect.y = hits[0].rect.bottom
                    self.facing = 'down'

    ''' Override Methods '''

    def update(self):
        if not self.transitioning:
            self.get_move()
        self.transition()

    def transition(self):
        scaled_x_change = math.floor(self.x_change / TRANSITION_SPEED)
        scaled_y_change = math.floor(self.y_change / TRANSITION_SPEED)
        if self.transition_loop >= TRANSITION_SPEED - 1:
            scaled_x_change += self.x_change % TRANSITION_SPEED
            scaled_y_change += self.y_change % TRANSITION_SPEED
        self.rect.x += scaled_x_change
        self.collide_blocks('x')
        self.rect.y += scaled_y_change
        self.collide_blocks('y')
        if self.transition_loop >= TRANSITION_SPEED - 1:
            self.transitioning = False
            self.transition_loop = 0

            self.x_change = 0
            self.y_change = 0
        elif self.transitioning:
            self.transition_loop += 1
        self.animate()

    def get_move(self):
        if self.game.auto:
            self.reference_move()
        else:
            self.movement()

    def reference_move(self):
        # Reference the game's character registry to get the next position
        positions = self.game.state.positions
        if self.id in positions.keys():
            curr_pos = positions[self.id]
            if len(self.game.state_history) >= 2:
                prev_state = self.game.state_history[-1]
                prev_pos = prev_state.positions[self.id]
                self.x_change += (curr_pos[0] - prev_pos[0]) * self.game.speeds["CHARACTER_SPEED"]
                self.y_change += (curr_pos[1] - prev_pos[1]) * self.game.speeds["CHARACTER_SPEED"]
                if self.x_change < 0:
                    self.facing = 'left'
                elif self.x_change > 0:
                    self.facing = 'right'
                if self.y_change < 0:
                    self.facing = 'up'
                elif self.y_change > 0:
                    self.facing = 'down'
                self.transitioning = True

    def movement(self):
        pass

    def set_animations(self):
        self.animations = None

class Enemy(Character):
    def __init__(self, game, x, y, id="U1"):
        Character.__init__(self, game, x, y, id=id, layer=game.layers["ENEMY_LAYER"], additional_groups=game.enemies)
        image = game.enemy_spritesheet.get_sprite(3, 2, self.width, self.height)
        self.set_image(image)
        self.facing = random.choice(['left', 'right'])
        self.max_travel = random.randint(7, 30)
        self.mobile = False

    def set_mobile(self, value):
        self.mobile = value

    def movement(self):
        if self.mobile:
            if self.facing == 'left':
                self.x_change -= game.speeds["ENEMY_SPEED"]
                self.movement_loop -= 1
                if self.movement_loop <= -self.max_travel:
                    self.facing = 'right'
            if self.facing == 'right':
                self.x_change += game.speeds["ENEMY_SPEED"]
                self.movement_loop += 1
                if self.movement_loop >= self.max_travel:
                    self.facing = 'left'

    def set_animations(self):
        down_animations = [self.game.enemy_spritesheet.get_sprite(3, 2, self.width, self.height),
                           self.game.enemy_spritesheet.get_sprite(35, 2, self.width, self.height),
                           self.game.enemy_spritesheet.get_sprite(68, 2, self.width, self.height)]

        up_animations = [self.game.enemy_spritesheet.get_sprite(3, 34, self.width, self.height),
                         self.game.enemy_spritesheet.get_sprite(35, 34, self.width, self.height),
                         self.game.enemy_spritesheet.get_sprite(68, 34, self.width, self.height)]

        left_animations = [self.game.enemy_spritesheet.get_sprite(3, 98, self.width, self.height),
                           self.game.enemy_spritesheet.get_sprite(35, 98, self.width, self.height),
                           self.game.enemy_spritesheet.get_sprite(68, 98, self.width, self.height)]

        right_animations = [self.game.enemy_spritesheet.get_sprite(3, 66, self.width, self.height),
                            self.game.enemy_spritesheet.get_sprite(35, 66, self.width, self.height),
                            self.game.enemy_spritesheet.get_sprite(68, 66, self.width, self.height)]

        self.animations = {
            "down" : down_animations,
            "up" : up_animations,
            "left" : left_animations,
            "right" : right_animations
        }

class Player(Character):
    def __init__(self, game, x, y, id="U2"):
        Character.__init__(self, game, x, y, id=id, layer=game.layers["PLAYER_LAYER"])
        image = game.character_spritesheet.get_sprite(3, 2, self.width, self.height)
        self.set_image(image)

    def collide_enemy(self):
        hits = pygame.sprite.spritecollide(self, self.game.enemies, False)

        if False and hits: # deletes player if it touches an enemy
            self.kill()
            self.game.playing = False

    def update(self):
        if not self.transitioning:
            self.get_move()
        self.collide_enemy()
        self.transition()

    def movement(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            for sprite in self.game.all_sprites: # centers the camera
                sprite.rect.x += self.game.speeds["PLAYER_SPEED"]
            self.x_change -= self.game.speeds["PLAYER_SPEED"]
            self.facing = 'left'
        elif keys[pygame.K_RIGHT]:
            for sprite in self.game.all_sprites:
                sprite.rect.x -= self.game.speeds["PLAYER_SPEED"]
            self.x_change += self.game.speeds["PLAYER_SPEED"]
            self.facing = 'right'
        elif keys[pygame.K_UP]:
            for sprite in self.game.all_sprites:
                sprite.rect.y += self.game.speeds["PLAYER_SPEED"]
            self.y_change -= self.game.speeds["PLAYER_SPEED"]
            self.facing = 'up'
        elif keys[pygame.K_DOWN]:
            for sprite in self.game.all_sprites:
                sprite.rect.y -= self.game.speeds["PLAYER_SPEED"]
            self.y_change += self.game.speeds["PLAYER_SPEED"]
            self.facing = 'down'

    def collide_blocks(self, direction):
        if direction == "x":
            hits = pygame.sprite.spritecollide(self, self.game.blocks, False) # checking if the rect of two sprites are overlapping, False = don't delete sprite when collide
            if hits:
                if self.x_change > 0:
                    self.rect.x = hits[0].rect.left - self.rect.width # hits[0] = wall we're colliding with
                    for sprite in self.game.all_sprites:
                        sprite.rect.x += self.game.speeds["PLAYER_SPEED"]
                if self.x_change < 0:
                    self.rect.x = hits[0].rect.right
                    for sprite in self.game.all_sprites:
                        sprite.rect.x -= self.game.speeds["PLAYER_SPEED"]
        if direction == "y":
            hits = pygame.sprite.spritecollide(self, self.game.blocks, False)
            if hits:
                if self.y_change > 0:
                    self.rect.y = hits[0].rect.top - self.rect.height
                    for sprite in self.game.all_sprites:
                        sprite.rect.y += self.game.speeds["PLAYER_SPEED"]
                if self.y_change < 0:
                    self.rect.y = hits[0].rect.bottom
                    for sprite in self.game.all_sprites:
                        sprite.rect.y -= self.game.speeds["PLAYER_SPEED"]

    def set_animations(self):
        down_animations = [self.game.character_spritesheet.get_sprite(3, 2, self.width, self.height),
                           self.game.character_spritesheet.get_sprite(35, 2, self.width, self.height),
                           self.game.character_spritesheet.get_sprite(68, 2, self.width, self.height)]

        up_animations = [self.game.character_spritesheet.get_sprite(3, 34, self.width, self.height),
                         self.game.character_spritesheet.get_sprite(35, 34, self.width, self.height),
                         self.game.character_spritesheet.get_sprite(68, 34, self.width, self.height)]

        left_animations = [self.game.character_spritesheet.get_sprite(3, 98, self.width, self.height),
                           self.game.character_spritesheet.get_sprite(35, 98, self.width, self.height),
                           self.game.character_spritesheet.get_sprite(68, 98, self.width, self.height)]

        right_animations = [self.game.character_spritesheet.get_sprite(3, 66, self.width, self.height),
                            self.game.character_spritesheet.get_sprite(35, 66, self.width, self.height),
                            self.game.character_spritesheet.get_sprite(68, 66, self.width, self.height)]

        self.animations = {
            "down" : down_animations,
            "up" : up_animations,
            "left" : left_animations,
            "right" : right_animations
        }

''' Other Objects '''
class Button:
    def __init__(self, x, y, width, height, fg, bg, content, fontsize):
        self.font = pygame.font.Font('./verdana/verdana.ttf', 32)
        self.content = content

        self.x = x
        self.y = y

        self.width = width
        self.height = height

        # foreground and background colors
        self.fg = fg
        self.bg = bg

        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(self.bg)
        self.rect = self.image.get_rect()

        self.rect.x = self.x
        self.rect.y = self.y

        self.text = self.font.render(self.content, True, self.fg)
        self.text_rect = self.text.get_rect(center=(self.width/2, self.height/2))
        self.image.blit(self.text, self.text_rect)

    def is_pressed(self, pos, pressed):
        if self.rect.collidepoint(pos):
            if pressed[0]:
                return True
            return False
        return False
