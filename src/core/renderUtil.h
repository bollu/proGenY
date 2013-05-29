#pragma once
#include "../include/SFML/Graphics.hpp"
#include "../include/Box2D/Box2D.h"

class renderUtil{
public:

	static sf::Shape *createShape(b2Shape *shape);
	static sf::Shape *createPolyShape(b2PolygonShape *shape);
	static sf::Shape *createCircleShape(b2CircleShape *shape);
};
