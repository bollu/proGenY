/*
#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../defines/renderingLayers.h"

class boundaryCreator : public objectCreator{
private:
	viewProcess *viewProc;

	float thickness;
	vector2 levelDim;

public:

	boundaryCreator(viewProcess *_viewProc) : viewProc(_viewProc){}


	void Init(vector2 levelDim, float thickness){
		this->thickness = thickness;
		this->levelDim = levelDim;
	}

	Object *createObject(vector2 playerInitPos) const{

		Object *boundaryObject = new Object("boundary");

		vector2 *pos = boundaryObject->getPrimitive<vector2>(
			Hash::getHash("position"));
		*pos = vector2(0,0);//levelDim * 0.5; //+ vector2(thickness, thickness);

		
		PhyData physicsData;
		RenderData render;

		physicsData.bodyDef.type = b2_staticBody;
		physicsData.collisionType = Hash::getHash("terrain");

		{
			//BOTTOM---------------------------------------------------------
			b2PolygonShape *bottom = new b2PolygonShape(); 
			vector2 bottomCenter = vector2(levelDim.x / 2.0, 0);//vector2(levelDim.x / 2.0, 0);
			bottom->SetAsBox(levelDim.x / 2.0, thickness, bottomCenter, 0);

			b2FixtureDef bottomFixtureDef;
			bottomFixtureDef.shape = bottom;
			bottomFixtureDef.friction = 1.0;

			physicsData.fixtureDef.push_back(bottomFixtureDef);

			sf::Shape *shape = renderUtil::createShape(bottom, 
				viewProc);
			shape->setFillColor(sf::Color::Blue);
			
			shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
			render.addRenderer(renderer);

		}

		{
			//TOP---------------------------------------------------------
			b2PolygonShape *top = new b2PolygonShape(); 
			vector2 topCenter = vector2(levelDim.x / 2.0, levelDim.y);//vector2(levelDim.x / 2.0, levelDim.y / 2.0);
			top->SetAsBox(levelDim.x / 2.0, thickness, topCenter, 0);

			b2FixtureDef topFixtureDef;
			topFixtureDef.shape = top;
			topFixtureDef.friction = 1.0;
			topFixtureDef.restitution = 0.0;

			physicsData.fixtureDef.push_back(topFixtureDef);


			sf::Shape *shape = renderUtil::createShape(top, 
				viewProc);
			shape->setFillColor(sf::Color::Blue);
			shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
			render.addRenderer(renderer);

		}

		
		{
			//LEFT---------------------------------------------------------
			b2PolygonShape *left = new b2PolygonShape(); 
			vector2 leftCenter = vector2(0, levelDim.y / 2.0);//vector2(0, levelDim.y / 2.0);
			left->SetAsBox(thickness, levelDim.y / 2.0, leftCenter, 0);

			b2FixtureDef leftFixtureDef;
			leftFixtureDef.shape = left;
			leftFixtureDef.friction = 1.0;

			physicsData.fixtureDef.push_back(leftFixtureDef);

			sf::Shape *shape = renderUtil::createShape(left, 
				viewProc);
			shape->setFillColor(sf::Color::Blue);
			
			shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
			render.addRenderer(renderer);
		}

		{
			//RIGHT---------------------------------------------------------
			b2PolygonShape *right = new b2PolygonShape(); 
			vector2 rightCenter = vector2(levelDim.x, levelDim.y / 2.0);
			right->SetAsBox(thickness, levelDim.y / 2.0, rightCenter, 0);

			b2FixtureDef rightFixtureDef;
			rightFixtureDef.shape = right;
			rightFixtureDef.friction = 1.0;

			physicsData.fixtureDef.push_back(rightFixtureDef);

			sf::Shape *shape = renderUtil::createShape(right, 
				viewProc);
			shape->setFillColor(sf::Color::Blue);
			shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
			render.addRenderer(renderer);
		}

		boundaryObject->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(physicsData));
		boundaryObject->addProp(Hash::getHash("RenderData"),
		 new Prop<RenderData>(render));

		return boundaryObject;
	};

};
*/