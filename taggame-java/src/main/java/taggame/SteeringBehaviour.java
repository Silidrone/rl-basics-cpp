package taggame;

import math.geom2d.Vector2D;

public interface SteeringBehaviour {
    Vector2D getVelocity(StaticInfo staticInfo, Vector2D currentVelocity);
}
