package taggame;

import math.geom2d.Point2D;
import math.geom2d.Vector2D;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.AbstractMap.SimpleEntry;
import java.util.stream.Collectors;

public class DumbTagSteering implements SteeringBehaviour {
    double safeDistanceThreshold = 0.2;
    double chasingCornerDistanceThreshold = 0.2;
    double cornerWeightMultiplier = 0.3;

    List<Point2D> corners;
    TagPlayer me;
    TagGame arena;
    float maxVelocity;

    public DumbTagSteering(TagPlayer me, TagGame arena, int width, int height, float maxVelocity) {
        corners = new ArrayList<>() {{
            add(new Point2D(0, 0));
            add(new Point2D(0, height));
            add(new Point2D(width, 0));
            add(new Point2D(width, height));
        }};
        final double bottomLeftTopRightDistance = Math.hypot(width, height);
        safeDistanceThreshold *= bottomLeftTopRightDistance;
        chasingCornerDistanceThreshold *= bottomLeftTopRightDistance;
        cornerWeightMultiplier *= bottomLeftTopRightDistance;

        this.me = me;
        this.arena = arena;
        this.maxVelocity = maxVelocity;
    }

    @Override
    public Vector2D getVelocity(StaticInfo staticInfo, Vector2D currentVelocity) {
        Vector2D desiredVelocity = new Vector2D(0, 0);
        boolean isTagged = me.isTagged();
        Point2D myPosition = me.getStaticInfo().getPos();
        ArrayList<TagPlayer> otherPlayers = arena.getPlayers().stream()
                .filter(p -> p != me)
                .collect(Collectors.toCollection(ArrayList::new));

        if (isTagged) {
            // If the player is tagged, chase the closest opponent
            TagPlayer targetOpponent = null;
            double closestDistance = Double.MAX_VALUE;

            for (TagPlayer opponent : otherPlayers) {
                Point2D opponentPosition = opponent.getStaticInfo().getPos();
                double distance = myPosition.distance(opponentPosition);

                double cornerWeight = Math.max(0, 1.0 - (opponentPosition.distance(getNearestCorner(opponent)) / chasingCornerDistanceThreshold));

                if (distance - cornerWeight * cornerWeightMultiplier < closestDistance) {
                    closestDistance = distance;
                    targetOpponent = opponent;
                }
            }

            if (targetOpponent != null) {
                desiredVelocity = new Vector2D(myPosition, targetOpponent.getStaticInfo().getPos()).normalize().times(maxVelocity);
            }
        } else {
            // If not tagged, flee from the tagged opponent
            TagPlayer taggedOpponent = otherPlayers.stream().filter(TagPlayer::isTagged).findFirst().orElse(null);

            if (taggedOpponent != null) {
                Point2D opponentPosition = taggedOpponent.getStaticInfo().getPos();
                List<SimpleEntry<Point2D, Double>> orderedCorners = orderCornersByDistance(taggedOpponent);
                double distanceToOpponent = myPosition.distance(opponentPosition);

                double fleeWeight = Math.max(0, (safeDistanceThreshold - distanceToOpponent) / safeDistanceThreshold);
                double seekWeight = 1.0 - fleeWeight;
                Point2D furthestCornerFromOpponent = orderedCorners.get(orderedCorners.size() - 1).getKey();

                Vector2D desiredDirection = new Vector2D(myPosition, furthestCornerFromOpponent).normalize();
                Vector2D fromMeToOpponent =  new Vector2D(myPosition, opponentPosition).normalize();
                desiredVelocity = desiredDirection.times(seekWeight).plus(fromMeToOpponent.times(-fleeWeight));

                desiredVelocity = desiredVelocity.normalize().times(maxVelocity);
            }
        }

        return desiredVelocity;
    }

    protected Point2D getNearestCorner(TagPlayer player) {
        return orderCornersByDistance(player).get(0).getKey();
    }

    protected List<SimpleEntry<Point2D, Double>> orderCornersByDistance(TagPlayer player) {
        List<SimpleEntry<Point2D, Double>> distanceEntries = new ArrayList<>();

        for (Point2D corner : corners) {
            double distance = new Point2D(player.getStaticInfo().getPos().x(), player.getStaticInfo().getPos().y()).distance(corner);
            distanceEntries.add(new SimpleEntry<>(corner, distance));
        }

        distanceEntries.sort(Comparator.comparingDouble(SimpleEntry::getValue));

        return distanceEntries;
    }
}
