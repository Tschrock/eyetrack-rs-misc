use nalgebra::MatrixXx2;
use rand::prelude::*;
use rand::Rng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RansacError {
    #[error("Not enough data points to fit an ellipse")]
    NotEnoughDataPoints,
    #[error("Failed to find a good fit")]
    FailedToFindGoodFit,
    #[error("Failed to invert matrix")]
    InversionFailed,
}

pub fn fit_rotated_ellipse_ransac<R>(
    data: &MatrixXx2<f64>,
    rng: &mut R,
    max_iterations: Option<usize>,
    samples_to_fit: Option<usize>,
    error_threshold: Option<f64>,
) -> Result<(f64, f64, f64, f64, f64), RansacError>
where
    R: Rng + ?Sized,
{
    let max_iterations = max_iterations.unwrap_or(5);
    let samples_to_fit = samples_to_fit.unwrap_or(10);
    let error_threshold = error_threshold.unwrap_or(80.0);

    let mut count_max = 0;
    let mut effective_sample: Option<MatrixXx2<f64>> = None;

    if data.nrows() < samples_to_fit {
        return Err(RansacError::NotEnoughDataPoints);
    }

    for _ in 0..max_iterations {
        let sample = MatrixXx2::from_rows(&data.row_iter().choose_multiple(rng, samples_to_fit));
        let (a, b, c, d, e, f) = rotated_ellipse_data(&sample)?;

        let ellipse_model =
            |x: f64, y: f64| (a * x.powi(2) + b * x * y + c * y.powi(2) + d * x + e * y + f);

        let rows = &data
        .row_iter()
        .filter(|row| ellipse_model(row[0], row[1]).abs() < error_threshold)
        .collect::<Vec<_>>();

        if rows.len() > count_max {
            count_max = rows.len();
            effective_sample = Some(MatrixXx2::from_rows(rows));
        }
    }

    match effective_sample {
        None => return Err(RansacError::FailedToFindGoodFit),
        Some(s) => {
            fit_rotated_ellipse(&s)
        }
    }
}

pub fn fit_rotated_ellipse(
    data: &MatrixXx2<f64>,
) -> Result<(f64, f64, f64, f64, f64), RansacError> {
    let (a, b, c, d, e, f) = rotated_ellipse_data(data)?;

    let theta = 0.5 * (b / (a - c)).atan();

    let cx = (2.0 * c * d - b * e) / (b.powi(2) - 4.0 * a * c);
    let cy = (2.0 * a * e - b * d) / (b.powi(2) - 4.0 * a * c);

    let cu = a * cx.powi(2) + b * cx * cy + c * cy.powi(2) - f;

    let w = (cu
        / (a * theta.cos().powi(2) + b * theta.cos() * theta.sin() + c * theta.sin().powi(2)))
    .sqrt();
    let h = (cu
        / (a * theta.sin().powi(2) - b * theta.cos() * theta.sin() + c * theta.cos().powi(2)))
    .sqrt();

    Ok((cx, cy, w, h, theta))
}

fn rotated_ellipse_data(
    data: &nalgebra::MatrixXx2<f64>,
) -> Result<(f64, f64, f64, f64, f64, f64), RansacError> {
    let xs = data.column(0);
    let ys = data.column(1);

    let j = nalgebra::MatrixXx5::from_row_iterator(
        data.nrows(),
        xs.iter()
            .zip(ys.iter())
            .flat_map(|(x, y)| [x * y, y * y, *x, *y, 1.0]),
    );

    let y = xs.map(|v| -1.0 * v * v);

    let p = (j.transpose() * &j)
        .try_inverse()
        .ok_or(RansacError::InversionFailed)?
        * j.transpose()
        * &y;

    let a = 1.0;
    let b = p[(0, 0)];
    let c = p[(1, 0)];
    let d = p[(2, 0)];
    let e = p[(3, 0)];
    let f = p[(4, 0)];

    Ok((a, b, c, d, e, f))
}
