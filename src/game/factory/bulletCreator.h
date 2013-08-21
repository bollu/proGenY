#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/bulletProcessor.h"
#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/renderUtil.h"
#include "../defines/renderingLayers.h"


class bulletCreator : public objectCreator
{
private:

	viewProcess *viewProc;
	bulletProp *bullet;
	float radius;
	const Hash *enemyCollision;


public:
	bulletCreator ( viewProcess *_viewProc ) : viewProc( _viewProc ),
		                                   radius( 0 ){}

	void setBulletData ( bulletProp *data ){
		this->bullet = data;
	}


	void setCollisionRadius ( float radius ){
		this->radius = radius;
	}


	Object *createObject ( vector2 _pos ) const {
		renderData render;
		phyProp *phy = new phyProp();
		Object *obj  = new Object( "bullet" );
		vector2 *pos = obj->getProp<vector2>(Hash::getHash( "position" ));
		*pos = _pos;
	


		//physics------------------------------------------------------------
		phy->collisionType = Hash::getHash( "bullet" );
		phy->bodyDef.type  = b2_dynamicBody;

		//phy->bodyDef.type = b2_kinematicBody;
		phy->bodyDef.bullet = false;

		b2CircleShape *shape = new b2CircleShape();

		shape->m_radius = this->radius;

		b2FixtureDef fixtureDef;

		fixtureDef.shape       = shape;
		fixtureDef.friction    = 0.0;
		fixtureDef.restitution = 0.0;
		fixtureDef.isSensor    = true;
		phy->fixtureDef.push_back( fixtureDef );

		//renderer------------------------------------
		sf::Shape *sfShape = renderUtil::createShape( shape, viewProc );

		sfShape->setFillColor( sf::Color::Red );

		shapeRenderNode *shapeRenderer = new shapeRenderNode( sfShape );

		render.addRenderer( shapeRenderer );

		//final---------------------------------
		obj->addProp( Hash::getHash( "renderData" ), new Prop< renderData > ( render ) );
		obj->addProp( Hash::getHash( "phyProp" ), phy);
		obj->addProp( Hash::getHash( "bulletProp" ), this->bullet);

		return (obj);
	} //createObject
};