#pragma once
#include "ParticleSystem.h"
#include "../IO/logObject.h"


struct Particle {
    vector2 v;
    vector2 a;
    vector2 s;


    float timeLeft = 0;
    float lifetime =  0;

    void Spawn(vector2 pos, vector2 vel, vector2 acc, float lifetime) {
        this->s = pos;
        this->v = vel;
        this->a = acc;
        
        this->timeLeft = this->lifetime = lifetime;
    }

    void Update(float dt) {
        this->v += a * dt;
        this->timeLeft -= dt;

        this->s += (v * dt) + 0.5 * a * dt * dt;
    }

    float getNormalizedTimeLeft() {
    	return this->timeLeft / this->lifetime;
    }

    bool isDead() {
        return this->timeLeft <= 0;
    };
};


ParticleSystem::ParticleSystem(ParticleProp prop) : prop_(prop) {
	assert(prop_.baseShape != NULL);
	particles_ = new Particle[prop_.particleCount];
	lastReleasedParticleTime_ = 0;

};

void ParticleSystem::SpawnParticle(Particle &particle) {
	float randomNum = (rand() % 1000) / 1000.0f;

	util::Angle angle = prop_.createAngle.getValue(randomNum); 
	vector2 vel = prop_.beginVel.getValue(randomNum);
	vector2 acc = prop_.acc.getValue(randomNum);

	float life =  prop_.particleLife;

	vector2 pos = nullVector;

	//if origin is absolute, set particle position to be the absolute curent position of the particle system.
	if (!prop_.relativeOrigin) {
		pos = vector2::cast(this->getPosition()); 
	}

	//point.position = sf::Vector2f(200, 200);
	particle.Spawn(pos, vel, acc, life);
}


void ParticleSystem::Update(float dt) {
	lastReleasedParticleTime_ += dt;

	for (int i = 0; i < prop_.particleCount; i++) {

		Particle &particle = particles_[i]; 

		// if the particle is dead, respawn it
		if (particle.isDead() && (lastReleasedParticleTime_ >= prop_.particleReleaseDelay) ){
			SpawnParticle(particle);
			lastReleasedParticleTime_ -= prop_.particleReleaseDelay;
		}

		//update the vertex array with info
		particle.Update(dt);
		
	}
};

sf::Color getColor(ParticleProp &prop, Particle &particle) {
	float t = 1.0f - particle.getNormalizedTimeLeft();
	sf::Color &begin = prop.beginColor;
	sf::Color &end = prop.endColor;

	return sf::Color( 
		util::lerp(begin.r, end.r, t),
		util::lerp(begin.g, end.g, t),
		util::lerp(begin.b, end.b, t),
		util::lerp(begin.a, end.a, t) );
}
void ParticleSystem::Draw(sf::RenderWindow &window) {
	//states.transform *= getTransform();

	for (int i = 0; i < prop_.particleCount; i++) {
		Particle &particle = particles_[i]; 
		
		if (particle.isDead()) {
			continue;
		}
		
		sf::Shape *shape = prop_.baseShape;

		sf::RenderStates renderMode = sf::RenderStates::Default;
		renderMode.blendMode = prop_.blendMode;
		
		//if the origin is relative, then we have to actually setup the relative origin
		if (prop_.relativeOrigin) {
			shape->setPosition(this->getPosition() + particle.s.cast<sf::Vector2f>());
		}
		else {
			shape->setPosition(particle.s.cast<sf::Vector2f>());
		}
		shape->setFillColor(getColor(prop_, particle));

		window.draw(*shape, renderMode);
	}
};

/*
void ParticleSystem::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    // apply the transform
	

    // our particles don't use a texture
	states.texture = NULL;

    // draw the vertex array
	target.draw(points_, states);
	
}
*/