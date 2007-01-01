#pragma once
#include "../../include/SFML/Graphics/Drawable.hpp"
#include "../../include/SFML/Graphics/Transformable.hpp"
#include "../../include/SFML/Graphics/VertexArray.hpp"
#include "Color.h"
#include "../controlFlow/Cooldown.h"


class TrailRenderer : public sf::Drawable, public sf::Transformable{
public:
	TrailRenderer(Cooldown<float> liveTime, Color beginColor, Color endColor);
	void Update(float dt);

private:
	Cooldown<float> liveTime_;
	Color begin_, end_;

	sf::VertexArray vertices;

	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
};