/*
 * This file is part of broccoli
 * Copyright (C) 2020 Chair of Applied Mechanics, Technical University of Munich
 * https://www.mw.tum.de/am
 */

#pragma once // Load this file only once

// This module requires Eigen library
#ifdef HAVE_EIGEN3

#include "../../geometry/rotations.hpp"
#include "FixedSizeQuaternionCurve.hpp"

namespace broccoli {
namespace curve {
    //! Class abstracting quaternion cuves using <b>S</b>pherical <b>QUAD</b>rangle curve interpolation (SQUAD) according to Shoemake 1987
    /*!
     * \ingroup broccoli_curve_curves
     *
     * \warning This class is restricted to **unit-quaternions**!
     *
     * The shape of the curve is described by a spherical quadrangle curve (from Boehm 1982):
     * \code
     *       q1-----------------q2
     *       /       xxxxx       \
     *      /    xxxx     xxxx    \
     *     /  xxx             xxx  \
     *    / xx                   xx \
     *   /xx                       xx\
     *  /x                           x\
     * q0                             q3
     * \endcode
     *
     * While \f$ q_0 \f$ and \f$ q_3 \f$ describe the control points (quaternions) at the beginning and end of the quadrangle, respectively, \f$ q_1 \f$ and \f$ q_2 \f$ are "virtual" control points used to design the shape of the curve.
     * Note that \f$ q_1 \f$ and \f$ q_2 \f$ may be used to interconnect segments in a smooth (\f$C^1\f$-continuous) way.
     *
     * \warning The original source [Shoemake 1987] is not available anymore. Instead [Dam 1998] is used as a reference for implementation.
     *
     * References
     * ----------
     * * Ken Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH Computer Graphics, ACM, New York, NY, USA, volume 19, number 3, 1985, DOI:[10.1145/325165.325242](https://www.doi.org/10.1145/325165.325242), p.245--254
     * * Ken Shoemake, "Quaternion calculus for animation", Math for SIGGRAPH (ACM SIGGRAPH ‘91 Course Notes #2), 1991.
     * * Wolfgang Boehm, "On Cubics: A Survey", Computer Graphics and Image Processing, 1982, volume 19, number 3, pages 201--226, DOI:[10.1016/0146-664X(82)90009-0](https://www.doi.org/10.1016/0146-664X(82)90009-0)
     * * Gerald Farin, "Curves and Surfaces for CAGD: A Practical Guide", Morgan Kaufmann Publishers, 2002, 5-th ed., ISBN: 1-55860-737-4
     * * Erik B. Dam et al., "Quaternions, Interpolation and Animation", Technical Report DIKU-TR-98/5, 1998, Department of Computer Science, University of Copenhagen, URL:[http://web.mit.edu/2.998/www/QuaternionReport1.pdf](http://web.mit.edu/2.998/www/QuaternionReport1.pdf)
     */
    class QuaternionSQUADCurve : public FixedSizeQuaternionCurve<4> {
    public:
        //! Specialized constructor
        /*!
         * \param [in] q0 The quaternion \f$ q_0 \f$ at the beginning of the curve (pass-through).
         * \param [in] q1 The first *virtual* control point \f$ q_1 \f$ used to define the shape of the curve (**no** pass-through)
         * \param [in] q2 The second *virtual* control point \f$ q_2 \f$ used to define the shape of the curve (**no** pass-through)
         * \param [in] q3 The quaternion \f$ q_3 \f$ at the end of the curve (pass-through).
         */
        QuaternionSQUADCurve(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, const Eigen::Quaterniond& q3)
        {
            // Set control points
            m_controlPoints[0] = q0;
            m_controlPoints[1] = q1;
            m_controlPoints[2] = q2;
            m_controlPoints[3] = q3;
        }

        //! Default constructor
        /*! Initializes as \f$ q_i = [w=1,\, x=0,\, y=0,\, z=0]^T \f$ for **all** control points */
        QuaternionSQUADCurve()
            : FixedSizeQuaternionCurve(Eigen::Quaterniond(1, 0, 0, 0))
        {
        }

        //! Destructor
        virtual ~QuaternionSQUADCurve()
        {
        }

        // Get type of underlying function (see base class for details)
        virtual FunctionType functionType() const { return FunctionType::SQUAD; }

        // Evaluation of (derivative of) underlying curve (see base class for details)
        //! \copydoc FixedSizeQuaternionCurve::evaluate()
        /*! \warning \p position has to within \f$[0,\,1]\f$, otherwise it will be projected! */
        virtual Eigen::Quaterniond evaluate(const double& position, const unsigned int& derivationOrder = 0) const
        {
            // Check validity
            assert(isValid());

            // Initialize return value
            Eigen::Quaterniond returnValue(1, 0, 0, 0);

            // Project interpolation parameter to boundaries
            double projectedPosition = position;
            if (projectedPosition < 0)
                projectedPosition = 0;
            if (projectedPosition > 1)
                projectedPosition = 1;

            // Check if we only want the base function (for speed up of computation)
            if (derivationOrder == 0) {
                /*! Evaluation of base function
                 *  ---------------------------
                 * \f[
                 * q(x) = SQUAD(q_0,\,q_1,\,q_2,\,q_3,\,x) = SLERP(SLERP(q_0,\,q_3,\,x),SLERP(q_1,\,q_2,\,x),2\,x(1-x))
                 * \f]
                 */
                // IMPORTANT: we do NOT use Eigen's implementation of slerp here since Eigen's implementation automatically chooses the shortest path. This is done through an if-else branch which can lead to discontinuities in special cases!
                const Eigen::Quaterniond slerp_q0q3 = geometry::quaternionSLERP(m_controlPoints[0], m_controlPoints[3], projectedPosition, false);
                const Eigen::Quaterniond slerp_q1q2 = geometry::quaternionSLERP(m_controlPoints[1], m_controlPoints[2], projectedPosition, false);
                returnValue = geometry::quaternionSLERP(slerp_q0q3, slerp_q1q2, 2.0 * projectedPosition * (1.0 - projectedPosition), false);
                returnValue.normalize(); // Normalize resulting quaternion to be sure to have a unit quaternion
            } else {
                /*! Evaluation of derivatives
                 *  -------------------------
                 * Computes **numeric** derivative with respect to the interpolation parameter \f$ x \f$ (**not** time in general).
                 * See \ref QuaternionCurve::evaluateNumericDerivative()
                 *
                 * \note There is no known closed form (analytic) solution for the \f$ n\f$-th derivative of this curve.
                 */
                returnValue = evaluateNumericDerivative(position, derivationOrder);
            }

            // Pass back return value
            return returnValue;
        }

        // Evaluation of value (D0) up to N-th derivative (DN) of underlying quaternion curve (see base class for details)
        /*!
         * \copydoc FixedSizeQuaternionCurve::evaluateD0ToDN()
         *
         * \warning \p position has to within \f$[0,\,1]\f$, otherwise it will be projected!
         */
        template <unsigned int N>
        std::array<Eigen::Quaterniond, N + 1> evaluateD0ToDN(const double& position) const
        {
            // Validity check
            assert(isValid());

            // Project interpolation parameter to boundaries
            double projectedPosition = position;
            if (projectedPosition < 0)
                projectedPosition = 0;
            if (projectedPosition > 1)
                projectedPosition = 1;

            // Trigger default implementation
            return FixedSizeQuaternionCurve::template evaluateD0ToDN<N>(projectedPosition);
        }

        // Encoding
        // --------
        // Encode member data as XML element and add it to the specified stream (see base class for details)
        virtual io::encoding::CharacterStreamSize encodeToXML(io::encoding::CharacterStream& stream, const size_t& XMLIndentationLevel, const size_t& XMLTabWhiteSpaces, const std::string& numericFormat = "%.7g") const
        {
            io::encoding::CharacterStreamSize addedElements = 0;

            // Start XML element
            for (size_t i = 0; i < XMLIndentationLevel * XMLTabWhiteSpaces; i++)
                addedElements += io::encoding::encode(stream, (char)' ');
            addedElements += io::encoding::encode(stream, "<QuaternionSQUADCurve");

            // Write attributes
            addedElements += encodeXMLAttributes(stream, numericFormat);

            // End XML element
            addedElements += io::encoding::encode(stream, "></QuaternionSQUADCurve>\n");

            // Pass back added elements in stream
            return addedElements;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // <-- Proper 128 bit alignment of member data necessary for Eigen vectorization
    };
} // namespace curve
} // namespace broccoli

#endif // HAVE_EIGEN3
