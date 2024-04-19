package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.ImageView;
import ij.ImagePlus;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

public final class ImageController {
    private final ImageModel imageModel;
    private final MainPanelController mainPanelController;
    private ImageView imageView;

    public ImageController(MainPanelController mainPanelController, ImageModel imageModel) {
        this.mainPanelController = mainPanelController;
        this.imageModel = imageModel;
        imageView = null;
    }

    public boolean isImageLoaded() {
        return imageModel.isImageLoaded();
    }

    public void loadImage(ImagePlus image, boolean simulation) {
        imageModel.loadImage(image, simulation);

        imageView = new ImageView();
        imageView.showImage(imageModel);

        image.getCanvas().addMouseListener(imageMouseClicked());
        image.getCanvas().addKeyListener(imageKeyPressed());

        imageModel.adapt_image_scale();

        mainPanelController.setLastFrame(imageModel.getStackSize());
    }

    public MouseListener imageMouseClicked() {
        return new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();

                System.out.println("x=" + x + " y=" + y);
            }

            @Override
            public void mousePressed(MouseEvent e) {

            }

            @Override
            public void mouseReleased(MouseEvent e) {

            }

            @Override
            public void mouseEntered(MouseEvent e) {

            }

            @Override
            public void mouseExited(MouseEvent e) {

            }
        };
    }

    public KeyListener imageKeyPressed() {
        return new KeyListener() {
            @Override
            public void keyTyped(KeyEvent e) {
            }

            @Override
            public void keyPressed(KeyEvent e) {
                System.out.println(e.getKeyCode());
            }

            @Override
            public void keyReleased(KeyEvent e) {
            }
        };
    }
}
