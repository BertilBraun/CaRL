import pygame
from typing import Optional


def get_infinite_line_intersection(
    p1: pygame.math.Vector2, p2: pygame.math.Vector2, p3: pygame.math.Vector2, p4: pygame.math.Vector2
) -> Optional[pygame.math.Vector2]:
    """
    Calculates the intersection point of two infinite lines.
    :return: A Vector2 of the intersection point, or None if lines are parallel.
    """
    den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    if abs(den) < 1e-6:
        return None

    t_num = (p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)
    t = t_num / den

    return pygame.math.Vector2(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))


def get_line_segment_intersection(
    p1: pygame.math.Vector2, p2: pygame.math.Vector2, p3: pygame.math.Vector2, p4: pygame.math.Vector2
) -> Optional[pygame.math.Vector2]:
    """
    Calculates the intersection point of two finite line segments.
    :return: A Vector2 of the intersection point, or None if they do not intersect.
    """
    den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    if den == 0:
        return None

    t_num = (p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)
    t = t_num / den

    u_num = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x))
    u = u_num / den

    if t > 0 and t < 1 and u > 0 and u < 1:
        return pygame.math.Vector2(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))

    return None
