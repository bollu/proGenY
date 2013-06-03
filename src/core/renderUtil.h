#pragma once
#include "../include/SFML/Graphics.hpp"
#include "../include/Box2D/Box2D.h"


class viewProcess;

class renderUtil{
public:

	static sf::Shape *createShape(b2Shape *shape, float game2RenderScale);
	static sf::Shape *createPolyShape(b2PolygonShape *shape, float game2RenderScale);
	static sf::Shape *createCircleShape(b2CircleShape *shape, float game2RenderScale);
};
