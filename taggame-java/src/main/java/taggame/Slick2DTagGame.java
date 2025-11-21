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
    protected static final Color PANEL_COLOR = new Color(45, 45, 50);  // Dark gray background
    protected static final Color SOFT_RED = new Color(220, 80, 80);     // Softer red
    protected static final Color SOFT_GREEN = new Color(76, 175, 80);   // Softer green
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
                    ? SOFT_GREEN
                    : SOFT_RED;
            p.render(graphics, color, tag_color);
        }

        // Draw labels in the flipped coordinate system
        for (int i = 0; i < players.size(); i++) {
            TagPlayer p = players.get(i);
            var pos = p.getStaticInfo().getPos();

            // Determine label based on player type
            String label = Objects.equals(p.getName(), rl_player_name)
                    ? "AI Agent"
                    : "Chaser";

            // Set text color to match player color
            Color textColor = Objects.equals(p.getName(), rl_player_name)
                    ? SOFT_GREEN
                    : SOFT_RED;

            graphics.setColor(textColor);

            // Save the current transform
            graphics.pushTransform();

            // Flip text back to be readable (since we're in flipped coordinate system)
            graphics.translate((float) pos.x(), (float) pos.y());
            graphics.scale(1, -1);

            // Draw label above the player
            float labelX = -30;  // Offset to center text approximately
            float labelY = -35;  // Above the player (negative because we flipped back)
            graphics.drawString(label, labelX, labelY);

            // Restore the transform
            graphics.popTransform();
        }

        graphics.scale(1, -1);
        graphics.translate(0, (float) -tagGame.getHeight());
    }

    @Override
    public int getID() {
        return id;
    }
}
