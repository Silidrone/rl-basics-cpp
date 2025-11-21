package taggame;

import math.geom2d.Point2D;
import math.geom2d.Vector2D;

public class Utils {
    public static Point2D toPoint(Vector2D v) {
        return new Point2D(v.x(), v.y());
    }

    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public static int[] toIntArray(Point2D point) {
        return new int[]{
                (int) (point.x() < 0 ? Math.floor(point.x()) : Math.ceil(point.x())),
                (int) (point.y() < 0 ? Math.floor(point.y()) : Math.ceil(point.y()))
        };
    }
}
