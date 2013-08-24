#pragma once
#include "../include/SFML/Graphics.hpp"
#include "../include/Box2D/Box2D.h"
#include "../core/vector.h"
#include "../core/Process/viewProcess.h"

class viewProcess;

/*!helps to convert box2d shapes into Render objects.

this is useful while debugging / prototyping. Can be used as placeholder art.
*/ 
class renderUtil{
public:

	/*!creates a Renderer shape from a box2d shape*/
	static sf::Shape *createShape(b2Shape *shape, viewProcess *view);
	static sf::Shape *createPolyShape(b2PolygonShape *shape, viewProcess *view);
	static sf::Shape *createCircleShape(b2CircleShape *shape, viewProcess *view);
	static sf::Shape *createRectangleShape(vector2 dim);
};
