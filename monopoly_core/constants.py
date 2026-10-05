"""
Official Monopoly Tournament Constants and Rules Specification.
Audited against Hasbro Official Tournament Rules and DARPA GNOME schema.
"""

from enum import IntEnum, auto

# Bank Inventory Limits (Strict Official Rules)
MAX_HOUSES = 32
MAX_HOTELS = 12
STARTING_CASH = 1500
GO_SALARY = 200
JAIL_FINE = 50
MAX_JAIL_TURNS = 3
INCOME_TAX = 200
LUXURY_TAX = 100
UNMORTGAGE_FEE_RATE = 0.10  # 10% interest to unmortgage

class TileType(IntEnum):
    SPECIAL = 0
    STREET = 1
    RAILROAD = 2
    UTILITY = 3
    TAX = 4

class ColorGroup(IntEnum):
    NONE = 0
    BROWN = 1
    SKY_BLUE = 2
    PINK = 3       # St. Charles, States, Virginia
    ORANGE = 4     # St. James, Tennessee, New York
    RED = 5        # Kentucky, Indiana, Illinois
    YELLOW = 6     # Atlantic, Ventnor, Marvin Gardens
    GREEN = 7      # Pacific, North Carolina, Pennsylvania Ave
    DARK_BLUE = 8  # Park Place, Boardwalk
    RAILROAD = 9
    UTILITY = 10

# Complete 40-square board metadata:
# (index, name, tile_type, color_group, price, mortgage, house_cost, base_rent, r1, r2, r3, r4, r_hotel)
BOARD_SPECS = [
    (0, "Go", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, "Mediterranean Avenue", TileType.STREET, ColorGroup.BROWN, 60, 30, 50, 2, 10, 30, 90, 160, 250),
    (2, "Community Chest 1", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (3, "Baltic Avenue", TileType.STREET, ColorGroup.BROWN, 60, 30, 50, 4, 20, 60, 180, 320, 450),
    (4, "Income Tax", TileType.TAX, ColorGroup.NONE, 0, 0, 0, 200, 0, 0, 0, 0, 0),
    (5, "Reading Railroad", TileType.RAILROAD, ColorGroup.RAILROAD, 200, 100, 0, 25, 50, 100, 200, 0, 0),
    (6, "Oriental Avenue", TileType.STREET, ColorGroup.SKY_BLUE, 100, 50, 50, 6, 30, 90, 270, 400, 550),
    (7, "Chance 1", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (8, "Vermont Avenue", TileType.STREET, ColorGroup.SKY_BLUE, 100, 50, 50, 6, 30, 90, 270, 400, 550),
    (9, "Connecticut Avenue", TileType.STREET, ColorGroup.SKY_BLUE, 120, 60, 50, 8, 40, 100, 300, 450, 600),
    (10, "Jail / Just Visiting", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (11, "St. Charles Place", TileType.STREET, ColorGroup.PINK, 140, 70, 100, 10, 50, 150, 450, 625, 750),
    (12, "Electric Company", TileType.UTILITY, ColorGroup.UTILITY, 150, 75, 0, 0, 0, 0, 0, 0, 0),
    (13, "States Avenue", TileType.STREET, ColorGroup.PINK, 140, 70, 100, 10, 50, 150, 450, 625, 750),
    (14, "Virginia Avenue", TileType.STREET, ColorGroup.PINK, 160, 80, 100, 12, 60, 180, 500, 700, 900),
    (15, "Pennsylvania Railroad", TileType.RAILROAD, ColorGroup.RAILROAD, 200, 100, 0, 25, 50, 100, 200, 0, 0),
    (16, "St. James Place", TileType.STREET, ColorGroup.ORANGE, 180, 90, 100, 14, 70, 200, 550, 750, 950),
    (17, "Community Chest 2", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (18, "Tennessee Avenue", TileType.STREET, ColorGroup.ORANGE, 180, 90, 100, 14, 70, 200, 550, 750, 950),
    (19, "New York Avenue", TileType.STREET, ColorGroup.ORANGE, 200, 100, 100, 16, 80, 220, 600, 800, 1000),
    (20, "Free Parking", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (21, "Kentucky Avenue", TileType.STREET, ColorGroup.RED, 220, 110, 150, 18, 90, 250, 700, 875, 1050),
    (22, "Chance 2", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (23, "Indiana Avenue", TileType.STREET, ColorGroup.RED, 220, 110, 150, 18, 90, 250, 700, 875, 1050),
    (24, "Illinois Avenue", TileType.STREET, ColorGroup.RED, 240, 120, 150, 20, 100, 300, 750, 925, 1100),
    (25, "B&O Railroad", TileType.RAILROAD, ColorGroup.RAILROAD, 200, 100, 0, 25, 50, 100, 200, 0, 0),
    (26, "Atlantic Avenue", TileType.STREET, ColorGroup.YELLOW, 260, 130, 150, 22, 110, 330, 800, 975, 1150),
    (27, "Ventnor Avenue", TileType.STREET, ColorGroup.YELLOW, 260, 130, 150, 22, 110, 330, 800, 975, 1150),
    (28, "Water Works", TileType.UTILITY, ColorGroup.UTILITY, 150, 75, 0, 0, 0, 0, 0, 0, 0),
    (29, "Marvin Gardens", TileType.STREET, ColorGroup.YELLOW, 280, 140, 150, 24, 120, 360, 850, 1025, 1200),
    (30, "Go To Jail", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (31, "Pacific Avenue", TileType.STREET, ColorGroup.GREEN, 300, 150, 200, 26, 130, 390, 900, 1100, 1275),
    (32, "North Carolina Avenue", TileType.STREET, ColorGroup.GREEN, 300, 150, 200, 26, 130, 390, 900, 1100, 1275),
    (33, "Community Chest 3", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (34, "Pennsylvania Avenue", TileType.STREET, ColorGroup.GREEN, 320, 160, 200, 28, 150, 450, 1000, 1200, 1400),
    (35, "Short Line Railroad", TileType.RAILROAD, ColorGroup.RAILROAD, 200, 100, 0, 25, 50, 100, 200, 0, 0),
    (36, "Chance 3", TileType.SPECIAL, ColorGroup.NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (37, "Park Place", TileType.STREET, ColorGroup.DARK_BLUE, 350, 175, 200, 35, 175, 500, 1100, 1300, 1500),
    (38, "Luxury Tax", TileType.TAX, ColorGroup.NONE, 0, 0, 0, 100, 0, 0, 0, 0, 0),
    (39, "Boardwalk", TileType.STREET, ColorGroup.DARK_BLUE, 400, 200, 200, 50, 200, 600, 1400, 1700, 2000),
]

# Color group members mapping
COLOR_GROUP_TILES = {
    ColorGroup.BROWN: [1, 3],
    ColorGroup.SKY_BLUE: [6, 8, 9],
    ColorGroup.PINK: [11, 13, 14],
    ColorGroup.ORANGE: [16, 18, 19],
    ColorGroup.RED: [21, 23, 24],
    ColorGroup.YELLOW: [26, 27, 29],
    ColorGroup.GREEN: [31, 32, 34],
    ColorGroup.DARK_BLUE: [37, 39],
    ColorGroup.RAILROAD: [5, 15, 25, 35],
    ColorGroup.UTILITY: [12, 28],
}

# 28 Purchasable property tile indices
ALL_PROPERTY_INDICES = [
    1, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 37, 39
]
