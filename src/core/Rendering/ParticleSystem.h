#pragma once
#include "../../include/SFML/Graphics/Drawable.hpp"
#include "../../include/SFML/Graphics/Transformable.hpp"
#include "../../include/SFML/Graphics/Shape.hpp"
#include "../../include/SFML/Graphics/RenderWindow.hpp"
#include "../../include/SFML/Graphics/RenderTarget.hpp"
#include "Color.h"
#include "../controlFlow/Cooldown.h"
#include "../math/Randomizer.h"
#include "../math/vector.h"
#include "../math/mathUtil.h"


struct Emitter {
    float radius = 0.0;
};

struct ParticleProp{
    sf::Shape *baseShape = NULL;

    Randomizer<util::Angle> createAngle = Randomizer<util::Angle>(util::Angle::Deg(0));
    Randomizer<vector2> beginVel = Randomizer<vector2>(nullVector);
    Randomizer<vector2> acc = nullVector;
    
    sf::Color beginColor;
    sf::Color endColor;

    float particleReleaseDelay = 0;
    float particleLife = 0;
    unsigned int particleCount = 0;

    sf::BlendMode blendMode =  sf::BlendMode::BlendAlpha;

    //whether the particles of the particle system must consider the particle system's as origin
    //__at all times__. If this is false, then particles of the particle system onc created will
    //consider SFML's origin as their origin, so they will not move along with the particle system
    bool relativeOrigin = false;
};

struct Particle;

class ParticleSystem : public sf::Transformable{
public:
    ParticleSystem(ParticleProp prop);
    void Update(float dt);
    void Draw(sf::RenderWindow &window);

private:
    float lastReleasedParticleTime_;

    Particle *particles_;
    ParticleProp prop_;

    void SpawnParticle(Particle &particle);

    //virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
};



