#pragma once
#include "renderUtil.h"
#include "vector.h"
#include "../util/logObject.h"
#include "../util/strHelper.h"

sf::Shape *renderUtil::createShape(b2Shape *shape, float game2RenderScale){

	switch(shape->m_type){
		case b2Shape::Type::e_polygon:
			return renderUtil::createPolyShape(static_cast<b2PolygonShape *>(shape), game2RenderScale);
			break;

		case b2Shape::Type::e_circle:
			return renderUtil::createCircleShape(static_cast<b2CircleShape *>(shape), game2RenderScale);

		default: 
			util::msgLog("unable to create required sf::Shape. unkown b2::shape",
				util::logLevel::logLevelError);
			return NULL;
	};
	
};


sf::Shape *renderUtil::createPolyShape(b2PolygonShape *b2Shape, float game2RenderScale){

	sf::ConvexShape *polyShape = new sf::ConvexShape(b2Shape->m_vertexCount);


	util::msgLog("vertex count: " + util::strHelper::toStr(b2Shape->m_vertexCount) );

	for(int i = 0; i < b2Shape->m_vertexCount; ++i){ 


		vector2 pt = vector2::cast<b2Vec2>(b2Shape->GetVertex(i)) * game2RenderScale;
		PRINTVECTOR2(pt);

		polyShape->setPoint(i, pt);
	}

	return polyShape;
};

sf::Shape *renderUtil::createCircleShape(b2CircleShape *b2Shape, float game2RenderScale){
	return new sf::CircleShape(b2Shape->m_radius * game2RenderScale * 0.5);
};