package taggame;

import math.geom2d.Vector2D;
import math.geom2d.Point2D;

public class StaticInfo {
    protected Point2D pos;
    protected double orientation;

    public StaticInfo(Point2D pos) {
        this.pos = pos;
        this.orientation = 0.0f;
    }

    public Point2D getPos() {
        return pos;
    }
    public double getOrientation() { return orientation; }

    public void update(Vector2D velocity, float time)
    {
        pos = pos.plus(velocity.times(time));
        if (velocity.norm() > 0.001)
            orientation = Math.atan2(velocity.y(),velocity.x());
    }
}
