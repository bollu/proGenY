#pragma once
#include "../include/SFML/Graphics.hpp"
#include "../include/Box2D/Box2D.h"
#include "../math/vector.h"
#include "viewProcess.h"

class viewProcess;

/*!helps to convert box2d shapes into Render objects.

this is useful while debugging / prototyping. Can be used as placeholder art.
*/ 
class renderUtil{
public:

	/*!creates a Renderer shape from a box2d shape*/
	static sf::Shape *createShape(const b2Shape *shape, viewProcess *view);
	static sf::Shape *createPolyShape(const b2PolygonShape *shape, viewProcess *view);
	static sf::Shape *createCircleShape(const b2CircleShape *shape, viewProcess *view);
	static sf::Shape *createRectangleShape(const vector2 &dim);
};
