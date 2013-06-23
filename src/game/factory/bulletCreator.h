#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/bulletProcessor.h"
#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/renderUtil.h"

class bulletCreator : public objectCreator{
private:
	viewProcess *viewProc;

	float radius;
	vector2 beginVel;

	const Hash *enemyCollision;

public:

	bulletCreator(viewProcess *_viewProc) : viewProc(_viewProc), radius(0){}

	void setRadius(float gRadius){
		this->radius = gRadius;
	}

	void setBeginVel(vector2 vel){
		this->beginVel = vel;
	}

	void setEnemyCollision(const Hash *enemyCollision){
		this->enemyCollision = enemyCollision;
	}

	Object *createObject(vector2 dummyPos) const{
		renderData render;
		phyData phy;
		bulletData bullet;
		
		Object *dummy = new Object("bullet");

		vector2 *pos = dummy->getProp<vector2>(Hash::getHash("position"));
		*pos = dummyPos;

		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("bullet");
		phy.bodyDef.type = b2_dynamicBody;

		
		b2CircleShape *dummyShape = new b2CircleShape();
		dummyShape->m_radius = this->radius;

		b2FixtureDef fixtureDef;
		fixtureDef.shape = dummyShape;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;
		fixtureDef.isSensor = true;

		phy.fixtureDef.push_back(fixtureDef);


		//renderer------------------------------------
		sf::Shape *shape = renderUtil::createShape(dummyShape, 
			viewProc->getGame2RenderScale());

		shape->setFillColor(sf::Color::Red);

		Renderer shapeRenderer(shape);
		render.addRenderer(shapeRenderer);
		
		//bullet-------------------------------------
		bullet.beginVel = this->beginVel;
		util::Angle angle = util::Angle::Rad(this->beginVel.toAngle());
		bullet.enemyCollision = this->enemyCollision;
		//TODO:add actual damage value
		bullet.damage = 1;

		//final---------------------------------
		dummy->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		dummy->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));
		dummy->addProp(Hash::getHash("bulletData"), 
			new Prop<bulletData>(bullet));
		
		return dummy;
	};
};