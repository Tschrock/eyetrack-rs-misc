//! Image processing module
//!
//! This module contains the image processing functions.

use opencv::core::{Mat, Point, Point2f, Rect, Scalar, Size, Vector};
use opencv::prelude::*;

#[allow(dead_code)]
pub fn crop_rotate(
    image: &Mat,
    x: i32,
    y: i32,
    height: i32,
    width: i32,
    angle: f64,
) -> Result<Mat, opencv::Error> {
    let rect = Rect::new(x, y, width, height);
    let cropped = Mat::roi(&image, rect)?;
    let size = cropped.size()?;
    let center = Point2f::new(width as f32 / 2.0, height as f32 / 2.0);
    let rotation_matrix = opencv::imgproc::get_rotation_matrix_2d(center, angle, 1.0)?;

    let mut rotated = Mat::default();
    opencv::imgproc::warp_affine(
        &cropped,
        &mut rotated,
        &rotation_matrix,
        size,
        opencv::imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        Scalar::all(255.0),
    )?;

    Ok(rotated)
}

pub fn greyscale(image: &Mat) -> Result<Mat, opencv::Error> {
    let mut grey = Mat::default();
    opencv::imgproc::cvt_color(&image, &mut grey, opencv::imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(grey)
}

#[allow(dead_code)]
pub fn circular_mask(image: &Mat, cx: i32, cy: i32, radius: i32) -> Result<Mat, opencv::Error> {
    let size = image.size()?;
    let mut mask = Mat::new_size_with_default(size, opencv::core::CV_8UC1, Scalar::all(0.0))?;

    opencv::imgproc::circle(
        &mut mask,
        Point::new(cx, cy),
        radius,
        Scalar::all(255.0),
        opencv::imgproc::FILLED,
        opencv::imgproc::LINE_8,
        0,
    )?;

    let mut masked_img = Mat::default();
    opencv::core::bitwise_and(image, image, &mut masked_img, &mask)?;

    let white = Mat::new_size_with_default(size, opencv::core::CV_8UC1, Scalar::all(255.0))?;

    let mask_inv = opencv::core::sub_scalar_mat(Scalar::all(255.0), &mask)?;

    let mut masked_white = Mat::default();
    opencv::core::bitwise_and(&white, &white, &mut masked_white, &mask_inv)?;

    let mut output = Mat::default();
    opencv::core::add(
        &masked_img,
        &masked_white,
        &mut output,
        &opencv::core::no_array(),
        -1,
    )?;

    Ok(output)
}

pub fn threshold(image: &Mat, threshold: f64) -> Result<Mat, opencv::Error> {
    let mut output = Mat::default();
    opencv::imgproc::threshold(
        &image,
        &mut output,
        threshold,
        255.0,
        opencv::imgproc::THRESH_BINARY,
    )?;
    Ok(output)
}

pub fn morphology(
    image: &Mat,
    kernel_width: i32,
    kernel_height: i32,
) -> Result<Mat, opencv::Error> {
    let kernel = opencv::imgproc::get_structuring_element(
        opencv::imgproc::MORPH_ELLIPSE,
        Size::new(kernel_width, kernel_height),
        Point::new(-1, -1),
    )?;

    let mut opening = Mat::default();
    opencv::imgproc::morphology_ex(
        &image,
        &mut opening,
        opencv::imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value()?,
    )?;

    let mut closing = Mat::default();
    opencv::imgproc::morphology_ex(
        &opening,
        &mut closing,
        opencv::imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value()?,
    )?;

    opencv::core::sub_scalar_mat(Scalar::all(255.0), &closing)?.to_mat()
}

pub fn find_contours(image: &Mat) -> Result<Vector<Mat>, opencv::Error> {
    let mut contours = opencv::types::VectorOfMat::new();
    opencv::imgproc::find_contours(
        &image,
        &mut contours,
        opencv::imgproc::RETR_TREE,
        opencv::imgproc::CHAIN_APPROX_NONE,
        Point::new(0, 0),
    )?;
    Ok(contours)
}

pub fn contours_to_convex_hulls(contours: &Vector<Mat>) -> Result<Vector<Vector<Point>>, opencv::Error> {
    let mut convex_hulls = opencv::types::VectorOfVectorOfPoint::new();
    for contour in contours {
        let mut hull = opencv::types::VectorOfPoint::new();
        opencv::imgproc::convex_hull(&contour, &mut hull, false, false)?;
        convex_hulls.push(hull);
    }
    Ok(convex_hulls)
}

pub fn find_largest_hull(hulls: &Vector<Vector<Point>>) -> Result<Option<Vector<Point>>, opencv::Error> {
    let mut largest_hull = None;
    let mut largest_hull_area = 0.0;
    for hull in hulls {
        // println!("hull: {:?}", hull);
        let area = opencv::imgproc::contour_area(&hull, false)?;
        if area > largest_hull_area {
            largest_hull = Some(hull);
            largest_hull_area = area;
        }
    }
    Ok(largest_hull)
}
