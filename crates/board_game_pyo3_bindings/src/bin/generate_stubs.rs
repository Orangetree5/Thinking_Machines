use _rust::stub_info;

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = stub_info()?;
    stub.generate()?;
    Ok(())
}
