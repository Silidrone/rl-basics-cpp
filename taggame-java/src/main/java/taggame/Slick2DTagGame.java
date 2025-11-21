package taggame;

import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

public class Slick2DTagGame extends BasicGameState {
    protected static final Color PANEL_COLOR = Color.black;
    protected static final ArrayList<Color> colors = new ArrayList<>(Arrays.asList(Color.green, Color.yellow, Color.magenta, Color.cyan, Color.orange, Color.pink));
    protected final TagGame tagGame;
    protected final int frame_rate;
    protected final Color rl_player_color;
    protected final Color tag_color;
    protected final int id;

    public Slick2DTagGame(String rl_player_name, Color rl_player_color, Color tag_color, int player_count, double player_radius, int width, int height, int frame_rate, float time_coefficient, float maxVelocity, int taggerSleepTimeMS, int id) {
        super();
        tagGame = new TagGame(rl_player_name, player_count, player_radius, width, height, time_coefficient, maxVelocity, taggerSleepTimeMS);
        this.frame_rate = frame_rate;
        this.rl_player_color = rl_player_color;
        this.tag_color = tag_color;
        this.id = id;
    }

    @Override
    public void init(GameContainer gameContainer, StateBasedGame stateBasedGame) {
        tagGame.initGame();
        gameContainer.setTargetFrameRate(frame_rate);
        gameContainer.setAlwaysRender(true);
        gameContainer.setVSync(false);
    }

    @Override
    public void update(GameContainer gameContainer, StateBasedGame stateBasedGame, int time) {
        try {
            tagGame.updateGame(time);
        } catch (RuntimeException e) {
            gameContainer.exit();
        }
    }

    @Override
    public void render(GameContainer gameContainer, StateBasedGame stateBasedGame, Graphics graphics) {
        graphics.clear();
        graphics.setColor(PANEL_COLOR);
        graphics.fillRect(0, 0, (float) tagGame.getWidth(), (float) tagGame.getHeight());

        graphics.translate(0, (float) tagGame.getHeight());
        graphics.scale(1, -1);

        var players = tagGame.getPlayers();
        var rl_player_name = tagGame.getRLPlayer().getName();
        for (int i = 0; i < players.size(); i++) {
            TagPlayer p = players.get(i);
            Color color = Objects.equals(p.getName(), rl_player_name)
                    ? this.rl_player_color
                    : colors.get(i % colors.size());
            p.render(graphics, color, tag_color);
        }

        graphics.scale(1, -1);
        graphics.translate(0, (float) -tagGame.getHeight());
    }

    @Override
    public int getID() {
        return id;
    }
}
