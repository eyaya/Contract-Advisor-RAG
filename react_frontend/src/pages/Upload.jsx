import Navbar from "../components/Navbar";
import { useNavigate} from "react-router-dom";
import { useState } from "react";
import aboutImg from "../assets/images/about.png";

function Upload() {
  
  const navigate = useNavigate();
  const [isUploading, setIsUploading] = useState(false);

  const handleFileSubmit = async (e) => {
    e.preventDefault();
    setIsUploading(true);
    
    const formData = new FormData(e.target);

    await fetch("upload_file", {
      method: "POST",
      body: formData,
    })
    .then((response) => response.json())
    .then(data => {
      console.log(data);
      alert(JSON.stringify(`${data.message}, status: ${data.status}`));
      navigate("/");
    })
    .catch((error) => {
      console.error("Error uploading the file:", error);
      });
      
      setIsUploading(false);
  };
  const info = [
    { text: "Years experience", count: "04" },
    { text: "Completed Projects", count: "24" },
    { text: "Companies Work", count: "06" },
  ];
  

  return (
    <div>
      <div>
      <Navbar />
      </div>
      <div className="container pt-5">
      <section id="about" className="py-10 text-white">
        <div className="text-center mt-8">
          <h3 className="text-4xl font-semibold">
            About <span className="text-cyan-600">Me</span>
          </h3>
          <p className="text-gray-400 my-3 text-lg">My introduction</p>
          <div className="flex md:flex-row flex-col-reverse items-center md:gap-6 gap-12 px-10 max-w-6xl mx-auto">
            <div className="p-2">
              <div className="text-gray-300 my-3">
                <p className="text-justify leading-7 w-11/12 mx-auto">
                  Lorem ipsum dolor sit amet consectetur adipisicing elit. Ipsa
                  eos, quam vel quisquam, explicabo sit labore dignissimos optio
                  ratione quibusdam doloribus pariatur consequuntur sint.
                  Reprehenderit cupiditate possimus facere quasi voluptatem?
                </p>
                <div className="flex mt-10 items-center gap-7">
                  {info.map((content) => (
                    <div key={content.text}>
                      <h3 className="md:text-4xl text-2xl font-semibold text-white">
                        {content.count}
                        <span className="text-cyan-600">+</span>{" "}
                      </h3>
                      <span className="md:text-base text-xs">{content.text}</span>
                    </div>
                  ))}
                </div>
                <br />
                <br />
                <a href="./src/assets/Code_a_program.pdf" download>
                  <button className="btn-primary">Download CV</button>
                </a>
              </div>
            </div>
            <div className="flex-1 md:mt-0 mt-6 flex justify-center items-center">
              <div className="lg:w-96 h-full relative sm:w-10/12 w-11/12 max-w-sm aboutImg ">
                <img
                  src={aboutImg}
                  alt=""
                  className="w-full object-cover bg-cyan-600 rounded-xl"
                />
              </div>
            </div>
          </div>
          <form encType="multipart/form-data" onSubmit={handleFileSubmit}>

          <div className="mb-3">
            <label htmlFor="formFile" className="form-label">
              Upload your file
            </label>
            <input
              name="file"
              className="form-control"
              type="file"
              id="formFile"
            />
          </div>
          <div className="form-group">
            <button className="btn btn-primary" type="submit" disabled={isUploading}>
              Submit
            </button>
          </div>
        </form>
        </div>
      </section>
      </div>
    </div>
  );
}

export default Upload;