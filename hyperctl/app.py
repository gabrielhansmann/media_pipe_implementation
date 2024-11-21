import curses

# Constants
TITLE = "Hyper-MoCap Controller"

# Utility functions

def draw_text(screen, y: int, x: int, text: str, attrs: list) -> None:
    with_attrs(screen, sum(attrs, 0), lambda: screen.addstr(y, x, text))

def with_attrs(screen, attrs: int, func):
    screen.attron(attrs)
    func()
    screen.attroff(attrs)

class LandingScreen:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.selected = 0

        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Default
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Status

        self.STYLE_DEFAULT = curses.color_pair(1)
        self.STYLE_STATUS = curses.color_pair(2)
    
    def draw(self):
        self.stdscr.clear()
        max_y, max_x = self.stdscr.getmaxyx()

        # Draw title
        title_x = (max_x - len(TITLE)) // 2
        draw_text(self.stdscr, 0, title_x, TITLE, [self.STYLE_DEFAULT])

        from devices import stringify_device_list
        for i, device in enumerate(stringify_device_list()):
            draw_text(self.stdscr, 2+i, 2, device, [self.STYLE_DEFAULT])

        self.stdscr.refresh()

    def handle_input(self):
        key = self.stdscr.getch()
        if key == ord('q'):
            return False
        return True

if __name__ == '__main__':
    def main(screen):
        app = LandingScreen(screen)
        try:
            while True:
                app.draw()
                if not app.handle_input():
                    break
        except KeyboardInterrupt:
            pass
    curses.wrapper(main)

