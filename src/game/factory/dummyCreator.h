#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/healthProcessor.h"
#include "../defines/renderingLayers.h"

class dummyCreator : public objectCreator{
private:
	viewProcess *viewProc;

	float radius;


public:

	dummyCreator(viewProcess *_viewProc) : viewProc(_viewProc), radius(0){}

	void setRadius(float gRadius){
		this->radius = gRadius;
	}

	Object *createObject(vector2 dummyPos) const{
		renderData render;
		phyData phy;
		healthData health;
		
		Object *dummy = new Object("dummy");


		vector2 *pos = dummy->getProp<vector2>(Hash::getHash("position"));
		*pos = dummyPos;

		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("dummy");
		phy.bodyDef.type = b2_dynamicBody;

		b2PolygonShape *dummyShape = new b2PolygonShape();
		dummyShape->SetAsBox(this->radius,
							this->radius);

		b2FixtureDef fixtureDef;
		fixtureDef.shape = dummyShape;
		fixtureDef.friction = 1.0;
		fixtureDef.restitution = 0.0;

		phy.fixtureDef.push_back(fixtureDef);


		//renderer------------------------------------
		sf::Shape *shape = renderUtil::createShape(dummyShape, 
			viewProc);

		shape->setFillColor(sf::Color::Red);

		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::HUD);
		render.addRenderer(renderer);
		
		//health-----------------------------------------
		health.setHP(10);

		//final---------------------------------
		dummy->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		dummy->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));
		dummy->addProp(Hash::getHash("healthData"), 
			new Prop<healthData>(health));
		
		return dummy;
	}
};