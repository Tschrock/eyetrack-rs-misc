use std::error::Error;
use std::path::Path;
use std::time::Instant;

mod image_processing;
mod ransac;
use opencv::prelude::*;

use opencv::core::Size;
use opencv::videoio::{
    VideoCapture, VideoWriter, CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
    // CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
};

const TEST_VIDEO: &str = "../assets/test.mp4";

// 320 × 240
const ROI_X: i32 = 40;
const ROI_Y: i32 = 20;
const ROI_W: i32 = 260;
const ROI_H: i32 = 160;

fn main() -> Result<(), Box<dyn Error>> {
    // Get the name and extension of the video
    let video_path = Path::new(TEST_VIDEO);
    let video_name = video_path.file_stem().unwrap().to_str().unwrap();
    // let video_ext = video_path.extension().unwrap();

    // Open the input video
    let mut input_video = VideoCapture::from_file(TEST_VIDEO, CAP_ANY)?;

    // Get the frames per second
    let fps = input_video.get(CAP_PROP_FPS)?;

    // Get the width and height of the video
    // let width = input_video.get(CAP_PROP_FRAME_WIDTH)?;
    // let height = input_video.get(CAP_PROP_FRAME_HEIGHT)?;

    // Get the number of frames
    let num_frames = input_video.get(CAP_PROP_FRAME_COUNT)?;

    // Define the codec
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;

    // Open the output videos
    let mut output_video = VideoWriter::new(
        format!("../assets/{}.out.mp4", video_name).as_str(),
        fourcc,
        fps,
        Size::new(ROI_W, ROI_H),
        true,
    )?;
    // let mut morph_video = VideoWriter::new(
    //     format!("../assets/{}.morph.mp4", video_name).as_str(),
    //     fourcc,
    //     fps,
    //     Size::new(ROI_W, ROI_H),
    //     false,
    // )?;

    // Setup pye3d
    let camera = eye3d::Camera {
        focal_length: 1.0,
        width: ROI_W as u32,
        height: ROI_H as u32,
    };
    let mut tracker3d = eye3d::EyeTracker3D::new(camera);

    let mut times = Vec::with_capacity(num_frames as usize);

    // Process the video
    let timer_video_start = Instant::now();
    let mut frame_number = 0.0;
    let mut image = Mat::default();
    while input_video.is_opened()? && input_video.read(&mut image)? {
        frame_number += 1.0;

        if frame_number < 120.0 {
            continue;
        }

        let timer_frame_start = Instant::now();
        let timer_cv_start = timer_frame_start;

        // ROI
        // println!("Processing frame {}", frame_number);
        // println!("ROI: {}, {}, {}, {}", ROI_X, ROI_Y, ROI_W, ROI_H);
        // println!("image size: {}, {}", image.cols(), image.rows());
        // let mut image = image_processing::crop_rotate(&image, ROI_X, ROI_Y, ROI_H, ROI_W, 0.0)?;

        let rect = opencv::core::Rect::new(ROI_X, ROI_Y, ROI_W, ROI_H);
        let mut image = Mat::roi(&image, rect)?;

        // Convert to grayscale
        let gray = image_processing::greyscale(&image)?;

        // Threshold
        let thres = image_processing::threshold(&gray, 70.0)?;

        // Morphology
        let morph = image_processing::morphology(&thres, 2, 2)?;

        let timer_cv_end = timer_cv_start.elapsed();

        // Debug morph video
        // morph_video.write(&morph)?;

        let timer_curves_start = Instant::now();

        // Contours
        let contours = image_processing::find_contours(&morph)?;

        // Hulls
        let hulls = image_processing::contours_to_convex_hulls(&contours)?;

        // Largest hull
        let largest_hull = match image_processing::find_largest_hull(&hulls)? {
            Some(v) => v,
            None => {
                output_video.write(&image)?;
                continue;
            }
        };

        let timer_curves_end = timer_curves_start.elapsed();
        let timer_ransac_start = Instant::now();

        // TODO: better way to convert opencv -> nalgebra?
        let hull_points = nalgebra::MatrixXx2::from_row_iterator(
            largest_hull.len(),
            largest_hull.iter().flat_map(|p| [p.x as f64, p.y as f64]),
        );

        // RANSAC hull points
        let (cx, cy, w, h, theta) = match ransac::fit_rotated_ellipse_ransac(
            &hull_points,
            &mut rand::thread_rng(),
            None,
            None,
            None,
        ) {
            Ok(v) => v,
            Err(_) => {
                output_video.write(&image)?;
                continue;
            }
        };

        let timer_ransac_end = timer_ransac_start.elapsed();

        // Draw the ellipse
        opencv::imgproc::ellipse(
            &mut image,
            opencv::core::Point2i::new(cx as i32, cy as i32),
            opencv::core::Size2i::new(w as i32, h as i32),
            theta,
            0.0,
            360.0,
            opencv::core::Vec4d::from_array([255.0, 0.0, 0.0, 255.0]),
            1,
            opencv::imgproc::LINE_4,
            0,
        )?;

        let timer_pye3d_start = Instant::now();
        let timer_pye3d_end;

        // pye3d
        if let Ok(result_3d) = tracker3d.process(
            frame_number / fps,
            eye3d::PupilEllipse {
                center: (cx, cy),
                major_radius: w,
                minor_radius: h,
                angle: theta,
            },
        ) {
            timer_pye3d_end = timer_pye3d_start.elapsed();

            // println!("{:?}", result_3d);
            // println!("----------");

            // Draw the projected circle
            let c = result_3d.eye_projected;
            opencv::imgproc::circle(
                &mut image,
                opencv::core::Point2i::new(c.center[0] as i32, c.center[1] as i32),
                c.radius.abs() as i32,
                opencv::core::Vec4d::from_array([0.0, 0.0, 255.0, 255.0]),
                1,
                opencv::imgproc::LINE_4,
                0,
            )?;

            // Draw the projected ellipse
            if let Some(e) = result_3d.pupil_projected {
                opencv::imgproc::ellipse(
                    &mut image,
                    opencv::core::Point2i::new(e.center[0] as i32, e.center[1] as i32),
                    opencv::core::Size2i::new(e.major_radius as i32, e.minor_radius as i32),
                    e.angle,
                    0.0,
                    360.0,
                    opencv::core::Vec4d::from_array([0.0, 255.0, 0.0, 255.0]),
                    1,
                    opencv::imgproc::LINE_4,
                    0,
                )?;
            }
        } else {
            timer_pye3d_end = timer_pye3d_start.elapsed();
        }

        let timer_frame_end = timer_frame_start.elapsed();

        times.push(nalgebra::RowVector5::new(
            timer_frame_end.as_secs_f64(),
            timer_cv_end.as_secs_f64(),
            timer_curves_end.as_secs_f64(),
            timer_ransac_end.as_secs_f64(),
            timer_pye3d_end.as_secs_f64(),
        ));

        // Write the frame
        // output_video.write(&image)?;
    }

    let timer_video_end = timer_video_start.elapsed().as_secs_f64();
    println!("Total time: {:.2}s", timer_video_end);
    println!("Average FPS: {:.2}", frame_number / timer_video_end);

    let times_avg = nalgebra::MatrixXx5::from_rows(times.as_slice()).row_mean();
    println!("Average time per frame: {:.3} ms", times_avg[0] * 1000.0);
    println!("    CV: {:.3} ms", times_avg[1] * 1000.0);
    println!("    Curves: {:.3} ms", times_avg[2] * 1000.0);
    println!("    RANSAC: {:.3} ms", times_avg[3] * 1000.0);
    println!("    eye3d: {:.3} ms", times_avg[4] * 1000.0);

    Ok(())
}
