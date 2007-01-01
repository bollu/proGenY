#pragma once
#include "TrailRenderer.h"
#include "../../include/SFML/Graphics/RenderTarget.hpp"
void TrailRenderer::draw(sf::RenderTarget& target, sf::RenderStates states) const {
	// apply the entity's transform -- combine it with the one that was passed by the caller
    states.transform *= getTransform(); // getTransform() is defined by sf::Transformable

    // apply the texture
    //states.texture = &m_texture;

    // you may also override states.shader or states.blendMode if you want

    // draw the vertex array
    target.draw(vertices, states);
}