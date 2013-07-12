#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/pickupProcessor.h"
#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/renderUtil.h"


class pickupCreator : public objectCreator{
private:
	viewProcess *viewProc;

	pickupData pickup;
	float radius;

public:

	pickupCreator(viewProcess *_viewProc) : viewProc(_viewProc), radius(0){}

	void setPickupData(pickupData data){
		this->pickup = data;
	}

	void setCollisionRadius(float radius){
		this->radius = radius;
	};

	
	Object *createObject(vector2 _pos) const{
		renderData render;
		phyData phy;
		
		Object *obj = new Object("bullet");

		vector2 *pos = obj->getProp<vector2>(Hash::getHash("position"));
		*pos = _pos;

		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("pickup");
		phy.bodyDef.type = b2_staticBody;

		
		b2CircleShape *shape = new b2CircleShape();
		shape->m_radius = this->radius;
		b2FixtureDef fixtureDef;
		fixtureDef.shape = shape;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;
		fixtureDef.isSensor = true;

		phy.fixtureDef.push_back(fixtureDef);


		//renderer------------------------------------
		float game2RenderScale = this->viewProc->getGame2RenderScale();
		sf::Shape *sfShape = new sf::CircleShape(this->radius * game2RenderScale,
										4);
		

		sfShape->setFillColor(sf::Color::Red);

		Renderer shapeRenderer(sfShape);
		render.addRenderer(shapeRenderer);
		
	
		//final---------------------------------
		obj->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		obj->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));
		obj->addProp(Hash::getHash("pickupData"), 
			new Prop<pickupData>(this->pickup));
		
		return obj;
	};
};