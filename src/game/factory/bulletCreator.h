#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/BulletProcessor.h"
#include "../../core/componentSys/processor/RenderProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"
#include "../../core/Rendering/renderUtil.h"
#include "../defines/renderingLayers.h"

/*

class bulletCreator : public objectCreator{
private:
	viewProcess *viewProc;

	BulletData bullet;
	float radius;

public:

	bulletCreator(viewProcess *_viewProc) : viewProc(_viewProc), radius(0){}

	void Init(BulletData data, float bulletRadius){
		this->bullet = data;
		this->radius = bulletRadius;
	}
	
	Object *createObject(vector2 _pos) const{
		RenderData render;
		PhyData phy;
		
		Object *obj = new Object("bullet");

		vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
		*pos = _pos;

		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("bullet");
		phy.bodyDef.type = b2_dynamicBody;
		//phy.bodyDef.type = b2_kinematicBody;
		phy.bodyDef.bullet = false;

		
		b2CircleShape *shape = new b2CircleShape();
		shape->m_radius = 1.0f;//this->radius;

		b2FixtureDef fixtureDef;
		fixtureDef.shape = shape;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;
		fixtureDef.isSensor = false;
		

		phy.fixtureDef.push_back(fixtureDef);


		//renderer------------------------------------
		sf::Shape *sfShape = renderUtil::createShape(shape, 
			viewProc);

		sfShape->setFillColor(sf::Color::Black);

		shapeRenderNode *shapeRenderer = new shapeRenderNode(sfShape, renderingLayers::action);
		render.addRenderer(shapeRenderer);
		
	
		//final---------------------------------
		obj->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(render));
		obj->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(phy));
		obj->addProp(Hash::getHash("BulletData"), 
			new Prop<BulletData>(this->bullet));
		
		return obj;
	};
};
*/