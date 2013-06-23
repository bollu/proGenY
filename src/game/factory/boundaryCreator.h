#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"

class boundaryCreator : public objectCreator{
private:
	viewProcess *viewProc;

	float thickness;
	vector2 levelDim;

public:

	boundaryCreator(viewProcess *_viewProc) : viewProc(_viewProc){}
	
	void setBoundaryThickness(float thickness){
		this->thickness = thickness;

	}

	void setDimensions(vector2 levelDim){
		this->levelDim = levelDim;
	}

	Object *createObject(vector2 playerInitPos) const{

		Object *boundaryObject = new Object("boundary");

		vector2 *pos = boundaryObject->getProp<vector2>(
			Hash::getHash("position"));
		*pos = levelDim * 0.5; //+ vector2(thickness, thickness);

		
		phyData physicsData;
		renderData render;

		physicsData.bodyDef.type = b2_staticBody;
		physicsData.collisionType = Hash::getHash("terrain");

		{
			//BOTTOM---------------------------------------------------------
			b2PolygonShape *bottom = new b2PolygonShape(); 
			vector2 bottomCenter = vector2(0, -levelDim.y / 2.0);//vector2(levelDim.x / 2.0, 0);
			bottom->SetAsBox(levelDim.x / 2.0, thickness, bottomCenter, 0);

			b2FixtureDef bottomFixtureDef;
			bottomFixtureDef.shape = bottom;
			bottomFixtureDef.friction = 1.0;

			physicsData.fixtureDef.push_back(bottomFixtureDef);

			sf::Shape *shape = renderUtil::createShape(bottom, 
				viewProc->getGame2RenderScale());
			shape->setFillColor(sf::Color::Blue);
			Renderer renderer(shape);
			render.addRenderer(renderer);

		}

		{
			//TOP---------------------------------------------------------
			b2PolygonShape *top = new b2PolygonShape(); 
			vector2 topCenter = vector2(0, levelDim.y / 2.0);//vector2(levelDim.x / 2.0, levelDim.y / 2.0);
			top->SetAsBox(levelDim.x / 2.0, thickness, topCenter, 0);

			b2FixtureDef topFixtureDef;
			topFixtureDef.shape = top;
			topFixtureDef.friction = 0.0;
			topFixtureDef.restitution = 0.0;

			physicsData.fixtureDef.push_back(topFixtureDef);


			sf::Shape *shape = renderUtil::createShape(top, 
				viewProc->getGame2RenderScale());
			shape->setFillColor(sf::Color::Blue);
			Renderer renderer(shape);
			render.addRenderer(renderer);

		}


		{
			//LEFT---------------------------------------------------------
			b2PolygonShape *left = new b2PolygonShape(); 
			vector2 leftCenter = vector2(-levelDim.x / 2.0, 0);
			left->SetAsBox(thickness, levelDim.y / 2.0, leftCenter, 0);

			b2FixtureDef leftFixtureDef;
			leftFixtureDef.shape = left;
			leftFixtureDef.friction = 1.0;

			physicsData.fixtureDef.push_back(leftFixtureDef);

			sf::Shape *shape = renderUtil::createShape(left, 
				viewProc->getGame2RenderScale());
			shape->setFillColor(sf::Color::Blue);
			Renderer renderer(shape);
			render.addRenderer(renderer);
		}

		{
			//RIGHT---------------------------------------------------------
			b2PolygonShape *right = new b2PolygonShape(); 
			vector2 rightCenter = vector2(levelDim.x / 2.0, 0);
			right->SetAsBox(thickness, levelDim.y / 2.0, rightCenter, 0);

			b2FixtureDef rightFixtureDef;
			rightFixtureDef.shape = right;
			rightFixtureDef.friction = 1.0;

			physicsData.fixtureDef.push_back(rightFixtureDef);

			sf::Shape *shape = renderUtil::createShape(right, 
				viewProc->getGame2RenderScale());
			shape->setFillColor(sf::Color::Blue);
			Renderer renderer(shape);
			render.addRenderer(renderer);
		}

		boundaryObject->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(physicsData));
		boundaryObject->addProp(Hash::getHash("renderData"),
		 new Prop<renderData>(render));

		return boundaryObject;
	};

};