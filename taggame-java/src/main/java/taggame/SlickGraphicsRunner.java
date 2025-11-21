package taggame;

import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.Bootstrap;

public class SlickGraphicsRunner extends StateBasedGame  {
    protected final static float TIME_COEFFICIENT = 1f;
    protected static final int WIDTH = 800;
    protected static final int HEIGHT = 800;
    protected static final float MAX_VELOCITY = 1;
    protected static final int FRAME_RATE = 60;
    protected static final int PLAYER_COUNT = 2;
    protected static final float PLAYER_RADIUS = 10;
    protected static final Color TAG_COLOR = Color.red;
    protected static final int TAGGER_SLEEP_TIME_MS = 1000;
    protected static final String RL_PLAYER_NAME = "Sili";
    protected static final Color RL_PLAYER_COLOR = Color.green;
    public static final String TAG_GAME = "Tag Game";

    protected static int tag_game_id = 0;

    public SlickGraphicsRunner() {
        super(TAG_GAME);
        addState(new Slick2DTagGame(RL_PLAYER_NAME, RL_PLAYER_COLOR, TAG_COLOR, PLAYER_COUNT,
                PLAYER_RADIUS, WIDTH, HEIGHT, FRAME_RATE, TIME_COEFFICIENT, MAX_VELOCITY,
                TAGGER_SLEEP_TIME_MS, tag_game_id));
    }
    @Override
    public void initStatesList(GameContainer gameContainer) {
        enterState(tag_game_id);
    }

    public static void main(String[] args) {
        SlickGraphicsRunner game = new SlickGraphicsRunner();
        Bootstrap.runAsApplication(game, WIDTH, HEIGHT, false);
    }
}
