package taggame;


import math.geom2d.Point2D;
import math.geom2d.Vector2D;
import org.newdawn.slick.Color;
import org.newdawn.slick.Graphics;

public class TagPlayer {
    public final static Vector2D noVelocity = new Vector2D(0, 0);
    public final static SteeringBehaviour idleSteering = (staticInfo, currentVelocity) -> noVelocity;

    protected StaticInfo staticInfo;
    protected Vector2D velocity = noVelocity;
    protected SteeringBehaviour steeringBehaviour = idleSteering;

    protected String name;
    protected boolean isTagged;
    protected double max_velocity;
    protected double radius;
    protected int game_width;
    protected int game_height;

    public TagPlayer(String name, StaticInfo staticInfo, double max_velocity,
                     double radius, int game_width, int game_height) {
        this.name = name;
        this.staticInfo = staticInfo;
        this.max_velocity = max_velocity;
        this.radius = radius;
        this.game_width = game_width;
        this.game_height = game_height;
    }

    public StaticInfo getStaticInfo() {
        return staticInfo;
    }

    public void setVelocity(Vector2D velocity) {
        if (velocity.norm() > max_velocity) {
            velocity = velocity.normalize().times(max_velocity);
        }

        this.velocity = velocity;
    }

    public Vector2D getVelocity() {
        return velocity;
    }

    public String getName() {return name;}

    public void setSteeringBehavior(SteeringBehaviour _steeringBehavior) {
        this.steeringBehaviour = _steeringBehavior;
    }

    public void update(float time) {
        if (steeringBehaviour != null) {
            setVelocity(steeringBehaviour.getVelocity(staticInfo, velocity));
        }

        Vector2D _velocity = new Vector2D(velocity.x(), velocity.y());
        
        // Only cancel out velocity components trying to move out of bounds
        // For x-axis
        if ((staticInfo.getPos().x() + this.radius >= this.game_width && velocity.x() > 0) || 
            (staticInfo.getPos().x() - this.radius <= 0 && velocity.x() < 0)) {
            _velocity = new Vector2D(0, _velocity.y());
        }
        
        // For y-axis
        if ((staticInfo.getPos().y() + this.radius >= this.game_height && velocity.y() > 0) || 
            (staticInfo.getPos().y() - this.radius <= 0 && velocity.y() < 0)) {
            _velocity = new Vector2D(_velocity.x(), 0);
        }

        staticInfo.update(_velocity, time);
    }

    public void render(Graphics graphics, Color color, Color tagColor) {
        Color c = graphics.getColor();
        var pos = staticInfo.getPos();
        double orientation = staticInfo.getOrientation();

        graphics.setColor(color);
        graphics.fillOval((float) ((float) pos.x() - this.radius), (float) ((float) pos.y() - this.radius), (float) (2 * this.radius), (float) (2 * this.radius));
        float x = (float) pos.x();
        float y = (float) pos.y();
        float ex = (float) (x + this.radius * Math.cos(orientation));
        float ey = (float) (y + this.radius * Math.sin(orientation));
        graphics.setColor(Color.black);
        graphics.drawLine(x, y, ex, ey);
        graphics.setColor(c);

        if (isTagged) {
            float tagRadius = (float) (this.radius + 2);
            Point2D center = staticInfo.getPos();
            graphics.setColor(tagColor);
            graphics.setLineWidth(3);
            graphics.drawOval((float) center.x() - tagRadius, (float) center.y() - tagRadius, 2 * tagRadius, 2 * tagRadius);
            graphics.setLineWidth(1);
        }

        graphics.setColor(c);
    }

    public boolean isTagging(TagPlayer player) {
        return (this.radius + this.radius >= staticInfo.getPos().distance(player.staticInfo.getPos()));
    }

    public void setIsTagged(boolean v) {
        this.isTagged = v;
    }

    boolean isTagged() {
        return isTagged;
    }
}
