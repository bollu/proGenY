#pragma once
#include "renderUtil.h"
#include "vector.h"
#include "../util/logObject.h"
#include "../util/strHelper.h"
#include "../util/mathUtil.h"
#include "../core/Process/viewProcess.h"

sf::Shape *renderUtil::createShape(const b2Shape *shape, viewProcess *view){

	switch(shape->m_type){
		case b2Shape::Type::e_polygon:
			return renderUtil::createPolyShape(static_cast<const b2PolygonShape *>(shape), view);
			break;

		case b2Shape::Type::e_circle:
			return renderUtil::createCircleShape(static_cast<const b2CircleShape *>(shape), view);

		default: 
			util::errorLog<<"unable to create required sf::Shape. unkown b2::shape"<<util::flush;
			return NULL;
	};
	
};


sf::Shape *renderUtil::createPolyShape(const b2PolygonShape *b2Shape, viewProcess *view){

	sf::ConvexShape *polyShape = new sf::ConvexShape(b2Shape->m_vertexCount);


	util::infoLog<<"vertex count: "<<b2Shape->m_vertexCount;

	for(int i = 0; i < b2Shape->m_vertexCount; ++i){ 

		vector2 gamePt = vector2::cast<b2Vec2>(b2Shape->GetVertex(i));

		vector2 viewPt = view->game2ViewCoord(gamePt);

		//it's like this: there are box2d LOCAL coordinates.
		//so, all we need to do is to scale the coordinates up (which is what view coordinates are)
		//and then flip the Y axis(to convert from SFML topDown to box2d bottomUp Y coordinates)
		vector2 renderPt = vector2(viewPt.x, -viewPt.y); //TODO: find out WHY box2d shape coordinates is the same as SFML 
									//coordinates- IE - top left corner is (0,0) +ve x axis is right
									//and +ve y axis is down. I have no freaking clue why it's this way


		PRINTVECTOR2(renderPt);

		polyShape->setPoint(i, renderPt);
	}
	

	return polyShape;
};

sf::Shape *renderUtil::createCircleShape(const b2CircleShape *b2Shape, viewProcess *view){
	static const int numPoints = 10;
	static const float angleDelta = util::TwoPI / numPoints;

	float game2RenderScale = view->getGame2RenderScale();
	float renderRadius = b2Shape->m_radius * game2RenderScale;

	sf::ConvexShape *polyShape = new sf::ConvexShape(numPoints);
	util::Angle angle;

	for(int i = 0; i < numPoints; i++){
		angle = util::Angle::Rad(i * angleDelta);
		polyShape->setPoint(i, angle.polarProjection(renderRadius));
	}

	
	
	return polyShape;
};


sf::Shape *renderUtil::createRectangleShape(const vector2 &dim){
	float w = dim.x;
	float h = dim.y;

	sf::ConvexShape *polyShape = new sf::ConvexShape(4);
	//top left
	polyShape->setPoint(0, vector2(0, 0));
	//top right
	polyShape->setPoint(1, vector2(w, 0));
	//bottom right
	polyShape->setPoint(2, vector2(w, h));
	//bottom left
	polyShape->setPoint(3, vector2(0, h));

	return polyShape;
};
