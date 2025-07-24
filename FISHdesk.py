import pygame
import numpy as np
import random
import sys
import os

try:
    import win32api
    import win32con
    import win32gui
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False

# --- 全局配置参数 ---
NUM_POINTS = 13000
TARGET_FPS = 800
DOT_SIZE = 1
PET_COLOR = (50, 40, 40)
TRANSPARENT_COLOR = (1, 1, 1)
PET_SPEED = 0.3
ROTATION_SPEED = 1.4
WANDER_STRENGTH = 0.0005


WALL_REPULSION_STRENGTH = 1 


class DesktopPet:
    def __init__(self):
        pygame.init()
        self.screen_info = pygame.display.Info()
        self.screen_width = self.screen_info.current_w
        self.screen_height = self.screen_info.current_h
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.NOFRAME)
        if IS_WINDOWS:
            hwnd = pygame.display.get_wm_info()["window"]
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                   win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
            win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*TRANSPARENT_COLOR), 0, win32con.LWA_COLORKEY)
            # 设置窗口为顶层窗口
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                 win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        else:
            self.screen.set_colorkey(TRANSPARENT_COLOR)
        self.clock = pygame.time.Clock()
        self.pet_x = self.screen_width / 2
        self.pet_y = self.screen_height / 2
        self.pet_angle = random.uniform(0, 2 * np.pi)
        self.pet_orientation_angle = self.pet_angle
        self.t = 0
        self.t_step = np.pi / 240
        i = np.arange(NUM_POINTS, 0, -1)
        self.x = i.astype(float)
        self.y = i / 235.0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        return True

    def update_state(self):
        self.pet_angle += random.uniform(-WANDER_STRENGTH, WANDER_STRENGTH)
        self.pet_x += PET_SPEED * np.cos(self.pet_orientation_angle)
        self.pet_y += PET_SPEED * np.sin(self.pet_orientation_angle)
        self.check_bounds()
        angle_diff = self.pet_angle - self.pet_orientation_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        self.pet_orientation_angle += angle_diff * ROTATION_SPEED
        self.t += self.t_step

    def draw(self):
        self.screen.fill(TRANSPARENT_COLOR)
        k = (4 + np.sin(self.y*2 - self.t) * 3) * np.cos(self.x/29)
        e = self.y/8 - 13
        d = np.sqrt(k**2 + e**2)
        q = 3*np.sin(k*2) + 0.3/(k + np.finfo(float).eps) + \
            np.sin(self.y/25)*k*(9+4*np.sin(e*9 - d*3 + self.t*2))
        c = d - self.t
        local_u = q + 30*np.cos(c) + 200
        local_v = q*np.sin(c) + 39*d - 220
        centered_u = local_u - 200
        centered_v = -local_v + 220
        angle_correction = -np.pi / 2 
        cos_o = np.cos(self.pet_orientation_angle + angle_correction)
        sin_o = np.sin(self.pet_orientation_angle + angle_correction)
        rotated_u = centered_u * cos_o - centered_v * sin_o
        rotated_v = centered_u * sin_o + centered_v * cos_o
        screen_u = rotated_u + self.pet_x
        screen_v = rotated_v + self.pet_y
        for i in range(NUM_POINTS):
            pygame.draw.circle(self.screen, PET_COLOR, (screen_u[i], screen_v[i]), DOT_SIZE)
        pygame.display.flip()

    def check_bounds(self):
        buffer = 200
        
        # 1. 获取宠物的当前“惯性”向量
        dx = np.cos(self.pet_angle)
        dy = np.sin(self.pet_angle)
        
        # 2. 检查所有墙壁，并施加排斥力
        # 使用独立的if，因为可能同时靠近多个墙壁
        repulsed = False
        if self.pet_x < buffer:
            dx += WALL_REPULSION_STRENGTH # 左墙施加向右的力
            repulsed = True
        if self.pet_x > self.screen_width - buffer:
            dx -= WALL_REPULSION_STRENGTH # 右墙施加向左的力
            repulsed = True
        if self.pet_y < buffer:
            dy += WALL_REPULSION_STRENGTH # 上墙施加向下的力
            repulsed = True
        if self.pet_y > self.screen_height - buffer:
            dy -= WALL_REPULSION_STRENGTH # 下墙施加向上的力
            repulsed = True

        # 3. 如果受到了任何排斥力，就计算新的目标方向
        if repulsed:
            # 新的目标方向是所有力合成后的向量方向
            self.pet_angle = np.arctan2(dy, dx)
        
        # 4. 强制将宠物位置拉回边界内，防止穿墙
        self.pet_x = np.clip(self.pet_x, buffer, self.screen_width - buffer)
        self.pet_y = np.clip(self.pet_y, buffer, self.screen_height - buffer)


    def run(self):
        print("Pygame 桌面宠物已启动！")
        print("按 'ESC' 键来退出。")
        running = True
        while running:
            running = self.handle_events()
            self.update_state()
            self.draw()
            self.clock.tick(TARGET_FPS)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    pet = DesktopPet()
    pet.run()
