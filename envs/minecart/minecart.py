from __future__ import print_function

import math
import random
import os
import json

import numpy as np
import scipy.stats
import sys
from gym import Env
from gym.spaces import Box, Discrete
from scipy.stats import norm


try:
    import cairocffi as cairo
    CAIRO = True
    print("Using cairocffi backend",file=sys.stderr)
except:
    print("Failed to import cairocffi, trying cairo...",file=sys.stderr)
    try:
        import cairo
        CAIRO = True
        print("Using cairo backend",file=sys.stderr)
    except:
        print("Failed to import cairo, trying pygame...",file=sys.stderr)
        try:
            import pygame
            CAIRO = False
            print("Using pygame backend",file=sys.stderr)
        except:
            print('Failed to import pygame, rendering should not be used!',file=sys.stderr)


EPS_SPEED = 0.001  # Minimum speed to be considered in motion
HOME_X = .0
HOME_Y = .0
HOME_POS = (HOME_X, HOME_Y)

ROTATION = 10
MAX_SPEED = 1.

FUEL_MINE = -.05
FUEL_ACC = -.025
FUEL_IDLE = -0.005

CAPACITY = 1

ACT_MINE = 0
ACT_LEFT = 1
ACT_RIGHT = 2
ACT_ACCEL = 3
ACT_BRAKE = 4
ACT_NONE = 5
ACTIONS = ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
ACTION_COUNT = len(ACTIONS)

MINE_RADIUS = 0.14
BASE_RADIUS = 0.15

WIDTH = 480
HEIGHT = 480

# Color definitions
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
C_GRAY = (150 / 255., 150 / 255., 150 / 255.)
DARK_GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
RED = (255, 70, 70)
C_RED = (1., 70 / 255., 70 / 255.)

FPS = 24

MINE_LOCATION_TRIES = 100

MINE_SCALE = 1.
BASE_SCALE = 1.
CART_SCALE = 1.

MARGIN = 0.16 * CART_SCALE

ACCELERATION = 0.0075 * CART_SCALE
DECELERATION = 1

CART_IMG = os.path.join(os.path.dirname(__file__), 'images', 'cart.png')
MINE_IMG = os.path.join(os.path.dirname(__file__), 'images', 'mine.png')

class Mine(object):
    """Class representing an individual Mine
    """

    def __init__(self, ore_cnt, x, y):
        self.distributions = [
            scipy.stats.norm(np.random.random(), np.random.random())
            for _ in range(ore_cnt)
        ]
        self.pos = np.array((x, y))

    def distance(self, cart):
        return mag(cart.pos - self.pos)

    def mineable(self, cart):
        return self.distance(cart) <= MINE_RADIUS * MINE_SCALE * CART_SCALE

    def mine(self):
        """Generates collected resources according to the mine's random
        distribution

        Returns:
            list -- list of collected ressources
        """

        return [max(0., dist.rvs()) for dist in self.distributions]

    def distribution_means(self):
        """
            Computes the mean of the truncated normal distributions
        """
        means = np.zeros(len(self.distributions))

        for i, dist in enumerate(self.distributions):
            mean, std = dist.mean(), dist.std()
            if np.isnan(mean):
                mean, std = dist.rvs(), 0
            means[i] = truncated_mean(mean,std,0,np.inf)

        return means


class Cart(object):
    """Class representing the actual minecart
    """

    def __init__(self, ore_cnt):
        self.ore_cnt = ore_cnt
        self.pos = np.array([HOME_X, HOME_Y])
        self.speed = 0
        self.angle = 45
        self.content = np.zeros(self.ore_cnt)
        self.departed = False  # Keep track of whether the agent has left the base

    def accelerate(self, acceleration):
        self.speed = clip(self.speed + acceleration, 0, MAX_SPEED)

    def rotate(self, rotation):
        self.angle = (self.angle + rotation) % 360

    def step(self):
        """
            Update cart's position, taking the current speed into account
            Colliding with a border at anything but a straight angle will cause
            cart to "slide" along the wall.
        """

        pre = np.copy(self.pos)
        if self.speed < EPS_SPEED:
            return False
        x_velocity = self.speed * math.cos(self.angle * math.pi / 180)
        y_velocity = self.speed * math.sin(self.angle * math.pi / 180)
        x, y = self.pos
        if y != 0 and y != 1 and (y_velocity > 0 + EPS_SPEED or
                                  y_velocity < 0 - EPS_SPEED):
            if x == 1 and x_velocity > 0:
                self.angle += math.copysign(ROTATION, y_velocity)
            if x == 0 and x_velocity < 0:
                self.angle -= math.copysign(ROTATION, y_velocity)
        if x != 0 and x != 1 and (x_velocity > 0 + EPS_SPEED or
                                  x_velocity < 0 - EPS_SPEED):
            if y == 1 and y_velocity > 0:
                self.angle -= math.copysign(ROTATION, x_velocity)

            if y == 0 and y_velocity < 0:
                self.angle += math.copysign(ROTATION, x_velocity)

        self.pos[0] = clip(x + x_velocity, 0, 1)
        self.pos[1] = clip(y + y_velocity, 0, 1)
        self.speed = mag(pre - self.pos)

        return True

class Minecart(Env):
    """Minecart environment
    """

    a_space = ACTION_COUNT

    def __init__(self,
                 mine_cnt=3,
                 ore_cnt=2,
                 capacity=CAPACITY,
                 mine_distributions=None,
                 ore_colors=None):

        super(Minecart, self).__init__()

        self.capacity = capacity
        self.ore_cnt = ore_cnt
        self.ore_colors = ore_colors or [(np.random.randint(
            40, 255), np.random.randint(40, 255), np.random.randint(40, 255))
            for i in range(self.ore_cnt)]

        self.mine_cnt = mine_cnt
        self.generate_mines(mine_distributions)
        self.cart = Cart(self.ore_cnt)

        self.end = False

        low = np.append(np.array([0, 0, 0, 0]), np.zeros(ore_cnt))
        high = np.append(np.array([1, 1, MAX_SPEED, 360]), np.ones(ore_cnt)*capacity)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Discrete(ACTION_COUNT)
        low = np.zeros(ore_cnt+1, dtype=np.float32)
        # assuming frameskip=4
        low[-1] = (FUEL_IDLE+FUEL_MINE)*4
        high = np.ones(ore_cnt+1)*capacity
        high[-1] = 0
        self.reward_space = Box(low=low, high=high, dtype=np.float32)

        self._initialized_render = False

    def _init_render(self):
        """initialize graphics backend, if there is rendering.
        """
        # Initialize graphics backend
        if CAIRO:
            self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, WIDTH,
                                              HEIGHT)
            self.context = cairo.Context(self.surface)
            self.initialized = True
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

            self.cart_sprite = pygame.sprite.Sprite()
            self.cart_sprites = pygame.sprite.Group()
            self.cart_sprites.add(self.cart_sprite)
            self.cart_image = pygame.transform.rotozoom(
                pygame.image.load(CART_IMG).convert_alpha(), 0,
                CART_SCALE)

            self.mine_sprites = pygame.sprite.Group()
            self.mine_rects = []
            for mine in self.mines:
                mine_sprite = pygame.sprite.Sprite()
                mine_sprite.image = pygame.transform.rotozoom(
                    pygame.image.load(MINE_IMG), mine.rotation,
                    MINE_SCALE).convert_alpha()
                self.mine_sprites.add(mine_sprite)
                mine_sprite.rect = mine_sprite.image.get_rect()
                mine_sprite.rect.centerx = (
                    mine.pos[0] * (1 - 2 * MARGIN)) * WIDTH + MARGIN * WIDTH
                mine_sprite.rect.centery = (
                    mine.pos[1] * (1 - 2 * MARGIN)) * HEIGHT + MARGIN * HEIGHT
                self.mine_rects.append(mine_sprite.rect)

    def obj_cnt(self):
        return self.ore_cnt + 1

    @staticmethod
    def from_json(filename):
        """
            Generate a Minecart instance from a json configuration file
            Args:
                filename: JSON configuration filename
        """
        with open(filename) as f:
            data = json.load(f)
        ore_colors = None if "ore_colors" not in data else data["ore_colors"]
        minecart = Minecart(
            ore_cnt=data["ore_cnt"],
            mine_cnt=data["mine_cnt"],
            capacity=data["capacity"],
            ore_colors=ore_colors)

        if "mines" in data:
            for mine_data, mine in zip(data["mines"], minecart.mines):
                mine.pos = np.array([mine_data["x"], mine_data["y"]])
                if "distributions" in mine_data:
                    mine.distributions = [
                        scipy.stats.norm(dist[0], dist[1])
                        for dist in mine_data["distributions"]
                    ]
            minecart.initialize_mines()
        return minecart

    def generate_mines(self, mine_distributions=None):
        """
            Randomly generate mines that don't overlap the base
            TODO: propose some default formations
        """
        self.mines = []
        for i in range(self.mine_cnt):
            pos = np.array((np.random.random(), np.random.random()))

            tries = 0
            while (mag(pos - HOME_POS) < BASE_RADIUS * BASE_SCALE + MARGIN) and (tries < MINE_LOCATION_TRIES):
                pos[0] = np.random.random()
                pos[1] = np.random.random()
                tries += 1
            assert tries < MINE_LOCATION_TRIES
            self.mines.append(Mine(self.ore_cnt, *pos))
            if mine_distributions:
                self.mines[i].distributions = mine_distributions[i]

        self.initialize_mines()

    def initialize_mines(self):
        """Assign a random rotation to each mine
        """

        for mine in self.mines:
            mine.rotation = np.random.randint(0, 360)

    def step(self, action, frame_skip=4):
        """Perform the given action `frame_skip` times
         ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
        Arguments:
            action {int} -- Action to perform, ACT_MINE (0), ACT_LEFT (1), ACT_RIGHT (2), ACT_ACCEL (3), ACT_BRAKE (4) or ACT_NONE (5)

        Keyword Arguments:
            frame_skip {int} -- Repeat the action this many times (default: {1})

        Returns:
            tuple -- (state, reward, terminal) tuple
        """
        change = False  # Keep track of whether the state has changed

        if action < 0 or action >= ACTION_COUNT:
            action = ACT_NONE

        reward = np.zeros(self.ore_cnt + 1)
        if frame_skip < 1:
            frame_skip = 1

        reward[-1] = FUEL_IDLE * frame_skip

        if action == ACT_ACCEL:
            reward[-1] += FUEL_ACC * frame_skip
        elif action == ACT_MINE:
            reward[-1] += FUEL_MINE * frame_skip
        for _ in range(frame_skip):

            if action == ACT_LEFT:
                self.cart.rotate(-ROTATION)
                change = True
            elif action == ACT_RIGHT:
                self.cart.rotate(ROTATION)
                change = True
            elif action == ACT_ACCEL:
                self.cart.accelerate(ACCELERATION)
            elif action == ACT_BRAKE:
                self.cart.accelerate(-DECELERATION)
            elif action == ACT_MINE:
                change = self.mine() or change

            if self.end:
                break

            change = self.cart.step() or change

            distanceFromBase = mag(self.cart.pos - HOME_POS)
            if distanceFromBase < BASE_RADIUS * BASE_SCALE:
                if self.cart.departed:
                    # Cart left base then came back, ending the episode
                    self.end = True
                    # Sell resources
                    reward[:self.ore_cnt] += self.cart.content
                    self.cart.content = np.zeros(self.ore_cnt)
            else:
                # Cart left base
                self.cart.departed = True

        # if not self.end and change:
        #     self.render()

        return self.get_state(change), reward.astype(np.float32), self.end, {}

    def mine(self):
        """Perform the MINE action

        Returns:
            bool -- True if something was mined
        """

        if self.cart.speed < EPS_SPEED:
            # Get closest mine
            mine = min(self.mines, key=lambda mine: mine.distance(self.cart))

            if mine.mineable(self.cart):
                cart_free = self.capacity - np.sum(self.cart.content)
                mined = mine.mine()
                total_mined = np.sum(mined)
                if total_mined > cart_free:
                    # Scale mined content to remaining capacity
                    scale = cart_free / total_mined
                    mined = np.array(mined) * scale

                self.cart.content += mined

                if np.sum(mined) > 0:
                    return True
        return False

    def get_pixels(self, update=True):
        """Get the environment's image representation

        Keyword Arguments:
            update {bool} -- Whether to redraw the environment (default: {True})

        Returns:
            np.array -- array of pixels, with shape (width, height, channels)
        """

        if update:
            if CAIRO:
                self.pixels = np.array(self.surface.get_data()).reshape(
                    WIDTH, HEIGHT, 4)[:, :, [2, 1, 0]]
            else:
                self.pixels = pygame.surfarray.array3d(self.screen)

        return self.pixels

    def get_state(self, update=True):
        """Returns the environment's full state, including the cart's position,
        its speed, its orientation and its content,

        Keyword Arguments:
            update {bool} -- Whether to update the representation (default: {True})

        Returns:
            np.ndarray -- array containing the aforementioned elements
        """

        return np.append(self.cart.pos,
                        [self.cart.speed, self.cart.angle, *self.cart.content]).astype(np.float32)

    def reset(self):
        """Reset's the environment to the start state

        Returns:
            [type] -- [description]
        """

        self.cart.content = np.zeros(self.ore_cnt)
        self.cart.pos = np.array(HOME_POS)
        self.cart.speed = 0
        self.cart.angle = 45
        self.cart.departed = False
        self.end = False

        return self.get_state()

    def __str__(self):
        string = "Completed: {} ".format(self.end)
        string += "Departed: {} ".format(self.cart.departed)
        string += "Content: {} ".format(self.cart.content)
        string += "Speed: {} ".format(self.cart.speed)
        string += "Direction: {} ({}) ".format(self.cart.angle,
                                               self.cart.angle * math.pi / 180)
        string += "Position: {} ".format(self.cart.pos)
        return string

    def render(self, mode='rgb_array'):
        """Update the environment's representation
        """
        # initalized graphics the first time there is rendering
        if not self._initialized_render:
            self._initialized_render = True
            self._init_render()
        if CAIRO:
            self.render_cairo()
        else:
            self.render_pygame()

        return self.get_pixels()

    def render_cairo(self):

        # Clear canvas
        self.context.set_source_rgba(*C_GRAY)
        self.context.rectangle(0, 0, WIDTH, HEIGHT)
        self.context.fill()

        # Draw home
        self.context.set_source_rgba(*C_RED)
        self.context.arc(HOME_X, HOME_Y,
                         int(WIDTH / 3 * BASE_SCALE), 0, 2 * math.pi)
        self.context.fill()

        # Draw Mines
        for mine in self.mines:
            draw_image(self.context, MINE_IMG,
                       (mine.pos[0] * (1 - 2 * MARGIN) + MARGIN) * WIDTH,
                       (mine.pos[1] * (1 - 2 * MARGIN) + MARGIN) * HEIGHT,
                       MINE_SCALE, -mine.rotation)

        # Draw cart
        cart_x = (self.cart.pos[0] * (1 - 2 * MARGIN) + MARGIN) * WIDTH
        cart_y = (self.cart.pos[1] * (1 - 2 * MARGIN) + MARGIN) * HEIGHT
        cart_surface = draw_image(self.context, CART_IMG, cart_x,
                                  cart_y, CART_SCALE, -self.cart.angle + 90)

        # Draw cart content
        width = cart_surface.get_width() / (2 * self.ore_cnt)
        height = cart_surface.get_height() / 3
        content_width = (width + 1) * self.ore_cnt
        offset = (cart_surface.get_width() - content_width) / 2
        for i in range(self.ore_cnt):

            rect_height = height * self.cart.content[i] / self.capacity

            if rect_height >= 1:

                self.context.set_source_rgba(*scl(self.ore_colors[i]))
                self.context.rectangle(cart_y - offset / 1.5,
                                       cart_x - offset + i * (width + 1),
                                       rect_height, width)
                self.context.fill()

    def render_pygame(self):

        pygame.event.get()
        # self.clock.tick(FPS)

        self.mine_sprites.update()

        # Clear canvas
        self.screen.fill(GRAY)

        # Draw Home
        pygame.draw.circle(self.screen, RED, (int(WIDTH * HOME_X), int(
            HEIGHT * HOME_Y)), int(WIDTH / 3 * BASE_SCALE))

        # Draw Mines
        self.mine_sprites.draw(self.screen)

        # Draw cart
        self.cart_sprite.image = rot_center(self.cart_image,
                                            -self.cart.angle).copy()

        self.cart_sprite.rect = self.cart_sprite.image.get_rect(
            center=(200, 200))

        self.cart_sprite.rect.centerx = self.cart.pos[0] * (
            1 - 2 * MARGIN) * WIDTH + MARGIN * WIDTH
        self.cart_sprite.rect.centery = self.cart.pos[1] * (
            1 - 2 * MARGIN) * HEIGHT + MARGIN * HEIGHT

        self.cart_sprites.update()

        self.cart_sprites.draw(self.screen)

        # Draw cart content
        width = self.cart_sprite.rect.width / (2 * self.ore_cnt)
        height = self.cart_sprite.rect.height / 3
        content_width = (width + 1) * self.ore_cnt
        offset = (self.cart_sprite.rect.width - content_width) / 2
        for i in range(self.ore_cnt):

            rect_height = height * self.cart.content[i] / self.capacity

            if rect_height >= 1:
                pygame.draw.rect(self.screen, self.ore_colors[i], (
                    self.cart_sprite.rect.left + offset + i * (width + 1),
                    self.cart_sprite.rect.top + offset * 1.5, width,
                    rect_height))

        pygame.display.update()


images = {}
def draw_image(ctx, image, top, left, scale, rotation):
    """Rotate, scale and draw an image on a cairo context
    """
    if image not in images:
        images[image] = cairo.ImageSurface.create_from_png(image)
    image_surface = images[image]
    img_height = image_surface.get_height()
    img_width = image_surface.get_width()
    ctx.save()
    w = img_height / 2
    h = img_width / 2

    left -= w
    top -= h

    ctx.translate(left + w, top + h)
    ctx.rotate(rotation * math.pi / 180.0)
    ctx.translate(-w, -h)

    ctx.set_source_surface(image_surface)

    ctx.scale(scale, scale)
    ctx.paint()
    ctx.restore()
    return image_surface


def rot_center(image, angle):
    """Rotate an image while preserving its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image



def mag(vector2d):
    return np.sqrt(np.dot(vector2d,vector2d))


def clip(val, lo, hi):
    return lo if val <= lo else hi if val >= hi else val


def scl(c):
    return (c[0] / 255., c[1] / 255., c[2] / 255.)


def truncated_mean(mean, std, a, b):
    if std == 0:
        return mean
    from scipy.stats import norm
    a = (a - mean) / std
    b = (b - mean) / std
    PHIB = norm.cdf(b)
    PHIA = norm.cdf(a)
    phib = norm.pdf(b)
    phia = norm.pdf(a)

    trunc_mean = (mean + ((phia - phib) / (PHIB - PHIA)) * std)
    return trunc_mean
