package taggame;

import math.geom2d.Point2D;
import math.geom2d.Vector2D;
import org.json.JSONObject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TagGame {
    protected static final int RL_PLAYER_INDEX = 0;

    protected final int player_count;
    protected final int width;
    protected final int height;
    protected final double maxDistance;
    protected final double player_radius;
    protected final float time_coefficient;
    protected final float maxVelocity;
    protected final int taggerSleepTimeMS;

    protected Communicator communicator;
    protected List<TagPlayer> players;
    protected TagPlayer tagPlayer;
    protected double tagChangedTime;
    protected String rl_player_name;

    protected Random rand;

    public TagGame(String rl_player_name, int player_count, double player_radius,
                   int width, int height, float time_coefficient, float maxVelocity, int taggerSleepTimeMS) {
        super();
        this.rl_player_name = rl_player_name;
        this.player_count = player_count;
        this.player_radius = player_radius;
        this.width = width;
        this.height = height;
        this.maxDistance = new Point2D(0, 0).distance(new Point2D(this.width, this.height)) - this.player_radius - 150;
        this.time_coefficient = time_coefficient;
        this.taggerSleepTimeMS = taggerSleepTimeMS;

        this.players = new ArrayList<>();
        this.tagChangedTime = 0;
        this.tagPlayer = null;
        this.maxVelocity = maxVelocity;
        this.rand = new Random();

        try {
            communicator = new Communicator();
        } catch (IOException e) {
            System.out.println("Failed to initialize communicator: " + e);
        }
    }

    public void initGame() {
        players.clear();
        tagPlayer = null;
        this.tagChangedTime = 0;

        for (int i = 0; i < player_count; i++) {
            TagPlayer player = new TagPlayer(
                    i == RL_PLAYER_INDEX ? rl_player_name : ("P" + (i + 1)),
                    getRandomPosition(),
                    maxVelocity,
                    player_radius,
                    width,
                    height
            );

            players.add(player);
        }

        setTag(getRandomNonRLPlayer());

        boolean invalidReset = false;
        for(TagPlayer player : players) {
            if(player != tagPlayer && tagPlayer.isTagging(player)) invalidReset = true;
        }

        if(invalidReset) {
            initGame();
        }
    }

    public void updateGame(int time) throws RuntimeException {
        try {
            TagPlayer RL_player = players.get(RL_PLAYER_INDEX);

            String action = communicator.receiveAction();
            if (action == null || action.equalsIgnoreCase(Communicator.EXIT)) return;

            if (action.equals(Communicator.RESET)) {
                initGame();
            } else {
                RL_player.setSteeringBehavior((StaticInfo staticInfo, Vector2D currentVelocity) ->
                        deserializeAction(action));
            }

            boolean taggerSleeping = System.currentTimeMillis() - tagChangedTime < this.taggerSleepTimeMS;
            if (!taggerSleeping) {
                tagPlayer.setSteeringBehavior(new DumbTagSteering(tagPlayer, this, width, height, (float) (this.maxVelocity * 0.8f)));
                handleTaggingLogic();
            }

            for (TagPlayer player : players) {
                player.update(time * this.time_coefficient);
            }

            var serializedState = getSerializedGameState(players.get(RL_PLAYER_INDEX)).toString();
            communicator.sendState(serializedState);
        } catch (IOException e) {
            try {
                communicator.close();
                throw new RuntimeException(e);
            } catch (IOException ex) {
                throw new RuntimeException(e);
            }
        }
    }

    public List<TagPlayer> getPlayers() {
        return players;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public TagPlayer getRLPlayer() {
        return this.players.get(RL_PLAYER_INDEX);
    }

    protected Vector2D deserializeAction(String action) {
        JSONObject json = new JSONObject(action);
        double x = json.getInt("x");
        double y = json.getInt("y");
        return new Vector2D(x, y);
    }
    protected JSONObject getSerializedGameState(TagPlayer me) {
        JSONObject gameState = new JSONObject();

        var mp = Utils.toIntArray(me.getStaticInfo().getPos());
        var mv = Utils.toIntArray(Utils.toPoint(me.getVelocity()));
        var tp = Utils.toIntArray(tagPlayer.getStaticInfo().getPos());
        var tv = Utils.toIntArray(Utils.toPoint(tagPlayer.getVelocity()));

        gameState.put("mp", mp);
        gameState.put("mv", mv);
        gameState.put("tp", tp);
        gameState.put("tv", tv);
        gameState.put("t", me == tagPlayer);

        return gameState;
    }


    protected void handleTaggingLogic() {
        for (TagPlayer player : players) {
            if (player != tagPlayer && tagPlayer.isTagging(player)) {
                setTag(player);
                tagPlayer.setSteeringBehavior(TagPlayer.idleSteering);
                tagChangedTime = System.currentTimeMillis();
                return;
            }
        }
    }

    protected void setTag(TagPlayer newPlayer) {
        tagPlayer = newPlayer;
        players.forEach((p) -> p.setIsTagged(p == newPlayer));
    }

    protected TagPlayer getRandomNonRLPlayer() {
        Random rand = new Random();
        TagPlayer randomPlayer;
        do {
            randomPlayer = players.get(rand.nextInt(players.size()));
        } while (players.indexOf(randomPlayer) == RL_PLAYER_INDEX);
        return randomPlayer;
    }

    protected StaticInfo getRandomPosition() {
        return new StaticInfo(new Point2D(rand.nextDouble() * (this.width), rand.nextDouble() * (this.height)));
    }
}
