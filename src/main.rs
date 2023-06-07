use std::error::Error;
use std::time::Instant;
use std::f64::consts::PI;

mod image_processing;
mod ransac;
use opencv::prelude::*;

use opencv::core::Size;
use opencv::videoio::{
    VideoCapture, VideoWriter, CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
};

fn main() -> Result<(), Box<dyn Error>> {
    // let no_params = opencv::core::Vector::new();

    // Hardcoded ROI for testing
    let x = 0;
    let y = 85;
    let w = 381;
    let h = 311;

    // Open input video
    let mut video = VideoCapture::from_file("./assets/test.mkv", CAP_ANY)?;

    // Get framerate and sizes
    let framerate = video.get(CAP_PROP_FPS as i32)?;
    let size = opencv::core::Size::new(
        video.get(CAP_PROP_FRAME_WIDTH as i32)? as i32,
        video.get(CAP_PROP_FRAME_HEIGHT as i32)? as i32,
    );
    let roi_size = Size::new(w, h);
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;

    // set up pye3d
    let camera = pye3d::camera::CameraModel {
        focal_length: 1.0,
        resolution: (size.width as usize, size.height as usize),
    };
    let mut detector_3d = pye3d::detector_3d::Detector3D::new(
        camera, None, None, None, None, None, None, None, None, None, None, None, None,
    )?;

    // Open output videos
    let mut out_vid = VideoWriter::new("./assets/out.mp4", fourcc, framerate, size, true)?;
    let mut morph_vid = VideoWriter::new("./assets/morph.mp4", fourcc, framerate, roi_size, true)?;

    // Process frames
    let start = Instant::now();
    let mut image = Mat::default();
    while video.read(&mut image)? {
        // Crop ROI
        let cropped = image_processing::crop_rotate(&image, x, y, h, w, 0.0)?;
        assert!(cropped.size()? == opencv::core::Size::new(w, h));

        // Convert to grayscale
        let gray = image_processing::greyscale(&cropped)?;

        // Circular mask
        // let gray = image_processing::circular_mask(&gray, 50, 50, 50)?;

        // Threshold
        let thres = image_processing::threshold(&gray, 30.0)?;

        // Morphology
        let morph = image_processing::morphology(&thres, 3, 3)?;

        // Debug - record morphology video
        let mut tmp = Mat::default();
        opencv::imgproc::cvt_color(&morph, &mut tmp, opencv::imgproc::COLOR_GRAY2BGR, 0)?;
        morph_vid.write(&tmp)?;

        // Contours
        let contours = image_processing::find_contours(&morph)?;

        // Hulls
        let hulls = image_processing::contours_to_convex_hulls(&contours)?;

        // Largest hull
        let largest_hull = match image_processing::find_largest_hull(&hulls)? {
            Some(v) => v,
            None => {
                out_vid.write(&image)?;
                continue;
            }
        };

        // RANSAC hull points
        // TODO: better way to convert opencv -> nalgebra?
        let hull_points = nalgebra::MatrixXx2::from_row_iterator(
            largest_hull.len(),
            largest_hull.iter().flat_map(|p| [p.x as f64, p.y as f64]),
        );

        let (cx, cy, w, h, theta) = match ransac::fit_rotated_ellipse_ransac(
            &hull_points,
            &mut rand::thread_rng(),
            None,
            None,
            None,
        ) {
            Ok(v) => v,
            Err(_) => {
                out_vid.write(&image)?;
                continue;
            }
        };

        // pye3d
        let pupil_datum = pye3d::detector_3d::PupilDatum {
            ellipse: pye3d::detector_3d::PupilDatumEllipse {
                center: (cx, cy),
                axes: (w, h),
                angle: theta * 180.0 / PI,
            },
            diameter: w,
            location: (cx, cy),
            confidence: 0.99,
            timestamp: start.elapsed().as_millis() as u64,
        };
        let result_3d = detector_3d.update_and_detect(&pupil_datum, &gray, None, None);

        println!("{:?}", result_3d);
        println!("----------");

        // Draw ellipse
        opencv::imgproc::ellipse(
            &mut image,
            opencv::core::Point2i::new(cx as i32 + x, cy as i32 + y),
            opencv::core::Size2i::new(w as i32, h as i32),
            theta,
            0.0,
            360.0,
            opencv::core::Vec4d::from_array([255.0, 0.0, 0.0, 255.0]),
            1,
            opencv::imgproc::LINE_4,
            0,
        )?;
        out_vid.write(&image)?;
    }
    Ok(())
}
