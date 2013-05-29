#pragma once
#include "renderUtil.h"
#include "vector.h"
#include "../util/logObject.h"
#include "../util/strHelper.h"

sf::Shape *renderUtil::createShape(b2Shape *shape){

	switch(shape->m_type){
		case b2Shape::Type::e_polygon:
			return renderUtil::createPolyShape(static_cast<b2PolygonShape *>(shape));
			break;

		case b2Shape::Type::e_circle:
			return renderUtil::createCircleShape(static_cast<b2CircleShape *>(shape));

		default: 
			util::msgLog("unable to create required sf::Shape. unkown b2::shape", util::logLevel::logLevelError);
			return NULL;
	};
	
};


sf::Shape *renderUtil::createPolyShape(b2PolygonShape *b2Shape){

	sf::ConvexShape *polyShape = new sf::ConvexShape(b2Shape->m_vertexCount);


	util::msgLog("vertex count: " + util::strHelper::toStr(b2Shape->m_vertexCount) );

	for(int i = 0; i < b2Shape->m_vertexCount; ++i){ 


		vector2 pt = vector2::cast<b2Vec2>(b2Shape->GetVertex(i));
		PRINTVECTOR2(pt);

		polyShape->setPoint(i, pt);
	}

	return polyShape;
};

sf::Shape *renderUtil::createCircleShape(b2CircleShape *b2Shape){
	return new sf::CircleShape(b2Shape->m_radius);
};