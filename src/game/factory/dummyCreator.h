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
		healthData health;
		
		Object *dummy = new Object("dummy");


		vector2 *pos = dummy->getPrimitive<vector2>(Hash::getHash("position"));
		*pos = dummyPos;

		//physics------------------------------------------------------------
		b2BodyDef bodyDef;
		bodyDef.type = b2_dynamicBody;

		b2PolygonShape dummyShape;
		dummyShape.SetAsBox(radius, radius);

		b2FixtureDef fixtureDef;
		fixtureDef.shape = &dummyShape;
		fixtureDef.friction = 1.0;
		fixtureDef.restitution = 0.0;

		PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef);
		phy.collisionType = Hash::getHash("dummy");


		//renderer------------------------------------
			sf::Shape *SFMLShape = renderUtil::createShape(&dummyShape, viewProc);
			SFMLShape->setFillColor(sf::Color::Red);

			_RenderNode renderNode(SFMLShape, renderingLayers::action);
			RenderData renderData = RenderProcessor::createRenderData(&renderNode);

		
		//health-----------------------------------------
		health.maxHP = 10;

		//final---------------------------------
		dummy->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(renderData));
		dummy->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(phy));
		dummy->addProp(Hash::getHash("healthData"), 
			new Prop<healthData>(health));
		
		return dummy;
	}
};
