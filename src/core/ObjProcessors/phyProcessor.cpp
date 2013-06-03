#pragma once
#include "phyProcessor.h"

void phyProcessor::onObjectAdd(Object *obj){


	phyData *data = obj->getProp<phyData>(Hash::getHash("phyData"));
	vector2* gamePos = obj->getProp<vector2>(Hash::getHash("position"));


	//it doesn't have either, so chill and move on
	if(data == NULL){
		return;
	}


	data->bodyDef.position = *gamePos;
	b2Body *body = world.CreateBody(&data->bodyDef);
	data->body = body;
	data->body->SetUserData(obj);


	for(auto it = data->fixtureDef.begin(); it != data->fixtureDef.end(); ++it){
		b2FixtureDef fixtureDef = *it;
		b2Fixture *fixture = body->CreateFixture(&fixtureDef);

		data->fixtures.push_back(fixture);
	}

	obj->addProp(Hash::getHash("impulse"), new v2Prop(vector2(0, 0)));
	obj->addProp(Hash::getHash("velocity"), new v2Prop(vector2(0, 0)));
}

void phyProcessor::Process(float dt){


	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		phyData *data = obj->getProp<phyData>(Hash::getHash("phyData"));


		if(data == NULL){
			continue;
		}
		
		//you're guarenteed to have the position property
		vector2* pos = obj->getProp<vector2>(Hash::getHash("position"));
		vector2 *impulse = obj->getProp<vector2>(Hash::getHash("impulse")); 
		vector2 *vel = obj->getProp<vector2>(Hash::getHash("velocity")); 

		data->body->ApplyLinearImpulse(*impulse, data->body->GetWorldCenter());
		*impulse = vector2(0, 0);

		*vel = vector2::cast(data->body->GetLinearVelocity());


		vector2 newPos = vector2::cast( data->body->GetPosition() );
		pos->x = newPos.x;
		pos->y = newPos.y;


	}
}

void phyProcessor::onObjectRemove(Object *obj){
	phyData *data = obj->getPtrProp<phyData>(Hash::getHash("phyData"));

	if(data != NULL){
		world.DestroyBody(data->body);
	}
}