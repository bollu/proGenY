#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/HealthProcessor.h"
#include "../defines/renderingLayers.h"

class dummyCreator : public objectCreator{
private:
	viewProcess *viewProc;

	float radius;


public:

	dummyCreator(viewProcess *_viewProc) : viewProc(_viewProc), radius(0){}

	void Init(float gRadius){
		this->radius = gRadius;
	}

	Object *createObject(vector2 dummyPos) const{
		RenderData render;
		PhyData phy;
		healthData health;
		
		Object *dummy = new Object("dummy");


		vector2 *pos = dummy->getPrimitive<vector2>(Hash::getHash("position"));
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
		dummy->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(render));
		dummy->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(phy));
		dummy->addProp(Hash::getHash("healthData"), 
			new Prop<healthData>(health));
		
		return dummy;
	}
};
